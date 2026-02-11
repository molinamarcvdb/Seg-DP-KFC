"""
Fundus Optic Disc/Cup Segmentation Dataset.

Merges 4 public fundus datasets: REFUGE (400), Drishti-GS (101), ORIGA (400),
RIM-ONE-r3 (159) = 1060 total images. Masks: 0=background, 128=cup, 255=disc.
Binary targets: "disc" (mask > 0), "cup" (mask == 128).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


FUNDUS_DEFAULT_ROOT = "./data/fundus"

FUNDUS_DATASETS = ["REFUGE", "Drishti-GS", "ORIGA", "RIM-ONE-r3"]


def _discover_fundus_images(root: str):
    """Discover all image/mask pairs across the 4 fundus datasets."""
    root = Path(root)
    pairs = []
    for ds_name in FUNDUS_DATASETS:
        ds_path = root / ds_name
        if not ds_path.exists():
            continue
        images_dir = ds_path / "images"
        masks_dir = ds_path / "masks"
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
    return pairs


class FundusSegDataset(Dataset):
    """
    Merged fundus optic disc/cup segmentation dataset.

    Supports binary segmentation of "disc" (cup + disc region, mask > 0)
    or "cup" (cup only, mask value == 128).
    """

    def __init__(
        self,
        root: str = FUNDUS_DEFAULT_ROOT,
        split: str = "train",
        image_size: int = 256,
        target: str = "disc",
        augment: bool = False,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.target = target
        self.augment = augment

        all_pairs = _discover_fundus_images(root)
        if len(all_pairs) == 0:
            raise FileNotFoundError(
                f"No fundus images found in {root}. Expected subdirectories: {FUNDUS_DATASETS}"
            )

        # 80/20 train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_pairs))
        split_idx = int(0.8 * len(all_pairs))

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.pairs = [all_pairs[i] for i in selected]

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        if augment:
            self.aug_hflip = transforms.RandomHorizontalFlip(p=0.5)
            self.aug_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # Extract single-channel mask from RGB
        mask_arr = np.array(mask_rgb)[:, :, 0]  # all channels equal
        if self.target == "disc":
            binary = (mask_arr > 0).astype(np.uint8) * 255
        elif self.target == "cup":
            binary = ((mask_arr > 64) & (mask_arr < 192)).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unknown target: {self.target}. Use 'disc' or 'cup'.")
        mask = Image.fromarray(binary, mode="L")

        if self.augment:
            seed_val = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed_val)
            image = self.aug_hflip(image)
            torch.manual_seed(seed_val)
            mask = self.aug_hflip(mask)

            torch.manual_seed(seed_val + 1)
            image = self.aug_vflip(image)
            torch.manual_seed(seed_val + 1)
            mask = self.aug_vflip(mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


def get_fundus_dataloaders(
    root: str = FUNDUS_DEFAULT_ROOT,
    image_size: int = 256,
    target: str = "disc",
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Create train and val dataloaders for fundus segmentation."""
    train_dataset = FundusSegDataset(
        root=root, split="train", image_size=image_size,
        target=target, augment=True,
    )
    val_dataset = FundusSegDataset(
        root=root, split="val", image_size=image_size,
        target=target, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Fundus Dataset...")
    dataset = FundusSegDataset(image_size=256, target="disc")
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
