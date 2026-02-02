#!/usr/bin/env python
"""
Train segmentation model with DP-SGD and optional AdaDPS preconditioning.

4 comparison methods:
  1. baseline        - No DP (upper bound)
  2. dp_no_precond   - DP-SGD without preconditioning
  3. dp_adadps_public - DP-SGD + preconditioner from public data
  4. dp_adadps_oracle - DP-SGD + preconditioner from private data (oracle)

Usage:
    # Train with specific method
    uv run python scripts/train.py --method dp_adadps_public

    # Override settings
    uv run python scripts/train.py --method dp_adadps_public dp.epsilon=1.0
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import yaml
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DRIVEDataset, STAREDataset, HRFDataset, MergedVesselDataset, get_training_augmentations, get_validation_augmentations
from src.models import create_model
from src.training import create_loss, create_preconditioner, DPTrainer, NonDPTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_method_config(config: dict, method: str) -> dict:
    """Apply method-specific settings from experiments.yaml."""
    experiments_path = Path(__file__).parent.parent / "configs" / "experiments.yaml"
    if experiments_path.exists():
        experiments = load_config(str(experiments_path))
        if method in experiments:
            method_cfg = experiments[method]
            for key, value in method_cfg.items():
                if isinstance(value, dict):
                    config.setdefault(key, {}).update(value)
                else:
                    config[key] = value
    return config


def update_config(config: dict, overrides: list) -> dict:
    """Apply command-line overrides (e.g., 'dp.epsilon=1.0')."""
    for override in overrides:
        if '=' not in override:
            continue
        key, value = override.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        d[keys[-1]] = value
    return config


def get_dataset(name: str, path: str, split: str, transform, fold: int = 0, n_folds: int = 5, train: bool = True):
    if name.upper() == "DRIVE":
        return DRIVEDataset(data_dir=path, split=split, transform=transform,
                           validation_fold=fold, n_folds=n_folds, train=train)
    elif name.upper() == "STARE":
        return STAREDataset(data_dir=path, transform=transform, validation_fold=fold, n_folds=n_folds, train=train)
    elif name.upper() == "HRF":
        return HRFDataset(data_dir=path, transform=transform, validation_fold=fold, n_folds=n_folds, train=train)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train DP Segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default=None,
                        choices=["baseline", "dp_no_precond", "dp_adadps_public", "dp_adadps_oracle"],
                        help="Experiment method (overrides config)")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    config = load_config(str(config_path))

    # Apply method-specific config
    if args.method:
        config = apply_method_config(config, args.method)

    # Apply CLI overrides
    config = update_config(config, args.overrides)

    # Print config
    print("=" * 60)
    print(f"Method: {config['experiment']['name']}")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))

    # Seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config['experiment']['device']
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Datasets
    data_cfg = config['data']
    train_transform = get_training_augmentations(data_cfg['image_size'])
    val_transform = get_validation_augmentations()

    private_cfg = data_cfg['private']

    # Check if private is a list of datasets (merged) or single dataset
    if isinstance(private_cfg, list):
        # Merged dataset: combine multiple datasets
        train_datasets = []
        val_datasets = []
        dataset_names = []
        for ds_cfg in private_cfg:
            train_ds = get_dataset(
                name=ds_cfg['name'], path=ds_cfg['path'], split='training',
                transform=train_transform, fold=data_cfg['validation_fold'],
                n_folds=data_cfg['n_folds'], train=True,
            )
            val_ds = get_dataset(
                name=ds_cfg['name'], path=ds_cfg['path'], split='training',
                transform=val_transform, fold=data_cfg['validation_fold'],
                n_folds=data_cfg['n_folds'], train=False,
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            dataset_names.append(ds_cfg['name'])

        train_dataset = MergedVesselDataset(train_datasets)
        val_dataset = MergedVesselDataset(val_datasets)
        private_name = "+".join(dataset_names)
    else:
        # Single dataset
        train_dataset = get_dataset(
            name=private_cfg['name'], path=private_cfg['path'], split='training',
            transform=train_transform, fold=data_cfg['validation_fold'],
            n_folds=data_cfg['n_folds'], train=True,
        )
        val_dataset = get_dataset(
            name=private_cfg['name'], path=private_cfg['path'], split='training',
            transform=val_transform, fold=data_cfg['validation_fold'],
            n_folds=data_cfg['n_folds'], train=False,
        )
        private_name = private_cfg['name']

    num_workers = data_cfg['num_workers']
    pin_memory = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_cfg['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"\nPrivate dataset: {private_name}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model_cfg = config['model']
    model = create_model(
        name=model_cfg['name'], in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'], features=model_cfg.get('features', [32, 64, 128, 256]),
    )
    print(f"\nModel: {model_cfg['name']} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Loss
    loss_cfg = config['loss']
    loss_fn = create_loss(name=loss_cfg['name'], dice_weight=loss_cfg.get('dice_weight', 0.5),
                          bce_weight=loss_cfg.get('bce_weight', 0.5))

    # Optimizer
    train_cfg = config['training']
    if train_cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'], momentum=0.9, weight_decay=train_cfg['weight_decay'])

    # Scheduler
    scheduler = None
    if train_cfg['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])
    elif train_cfg['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['epochs'] // 3, gamma=0.1)

    # Output dir
    log_cfg = config['logging']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(log_cfg['output_dir']) / f"{config['experiment']['name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # DP or Non-DP training
    dp_cfg = config['dp']
    precond_cfg = config['preconditioning']

    if dp_cfg['enabled']:
        print(f"\n=== DP-SGD Training (ε={dp_cfg['epsilon']}) ===")
        print(f"Preconditioning: {precond_cfg['type']}")

        # Create preconditioner
        preconditioner = create_preconditioner(
            precond_type=precond_cfg['type'], damping=precond_cfg['damping'], device=device,
        )

        # Estimate preconditioner if needed
        if precond_cfg['type'] in ['adadps_public', 'adadps_oracle']:
            from opacus import GradSampleModule

            if precond_cfg['type'] == 'adadps_public':
                # Use public data for estimation
                public_cfg = data_cfg['public']
                public_dataset = get_dataset(
                    name=public_cfg['name'], path=public_cfg['path'],
                    split=public_cfg.get('split', 'training'), transform=train_transform,
                )
                estimation_loader = torch.utils.data.DataLoader(
                    public_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
                )
                print(f"\nEstimating preconditioner from PUBLIC data ({public_cfg['name']})...")
            else:
                # Oracle: use private data (NOT privacy-preserving!)
                estimation_loader = train_loader
                print(f"\nEstimating preconditioner from PRIVATE data (Oracle)...")

            # Wrap model for preconditioner estimation, then keep it wrapped
            model = GradSampleModule(model).to(device)
            preconditioner.estimate_from_loader(
                model=model, data_loader=estimation_loader,
                loss_fn=loss_fn, num_steps=precond_cfg['estimation_steps'],
            )
            preconditioner.save(str(output_dir / "preconditioner.pt"))
            print("Preconditioner estimated and saved.")
            # Model stays wrapped for DPTrainer

        # Create DP trainer
        trainer = DPTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, loss_fn=loss_fn,
            epsilon=dp_cfg['epsilon'], delta=dp_cfg['delta'],
            max_grad_norm=dp_cfg['max_grad_norm'], epochs=train_cfg['epochs'],
            device=device, preconditioner=preconditioner, scheduler=scheduler,
        )
    else:
        print(f"\n=== Baseline Training (No DP) ===")
        trainer = NonDPTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, loss_fn=loss_fn, epochs=train_cfg['epochs'],
            device=device, scheduler=scheduler,
        )

    # Checkpoint callback
    def save_checkpoint(epoch, model):
        if (epoch + 1) % log_cfg['save_every'] == 0:
            path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            state = model.state_dict() if not hasattr(model, '_module') else model._module.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state}, path)

    # Train
    print(f"\nOutput: {output_dir}")
    history = trainer.fit(save_fn=save_checkpoint)

    # Save final model and history
    final_model = trainer.model._module if hasattr(trainer.model, '_module') else trainer.model
    torch.save(final_model.state_dict(), output_dir / "model_final.pt")
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best Dice: {max(history['val_dice']):.4f}")


if __name__ == "__main__":
    main()
