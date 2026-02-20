#!/usr/bin/env python3
"""
Generate a figure showing BraTS segmentation predictions for each method.

Layout: 4 rows (cases) × N columns (Image, Ground Truth, method1, method2, ...)
Each cell shows an axial slice through the center of the tumor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from src.data.brats_dataset import BraTSPatchDataset, _load_volume_npz, _binarize_target
from src.models.unet3d import create_model_3d


# ---------- config ----------
DATA_ROOT = "data/brats_preprocessed"
PATCH_SIZE = 64
SEED = 42  # use seed 42 models
N_CASES = 4
THRESHOLD = 0.5
TARGET = "multilabel"
OUT_CHANNELS = 3
CHANNEL_NAMES = ["ET", "TC", "WT"]

# Methods and their model paths (seed=42 runs)
METHODS = {
    "DP-SGD": f"outputs/brats_dp_eps8.0_s42_20260217_181256/model_best.pt",
    "AdaDPS (syn)": f"outputs/brats_dp_synthetic_eps8.0_s42_20260217_184358/model_best.pt",
    "AdaDPS (oracle)": f"outputs/brats_dp_public_eps8.0_s42_20260217_191529/model_best.pt",
    "Shampoo (syn)": f"outputs/brats_dp_shampoo_synth_eps8.0_s42_20260217_202005/model_best.pt",
    "Shampoo (oracle)": f"outputs/brats_dp_shampoo_eps8.0_s42_20260217_194745/model_best.pt",
}


def _sliding_window_single(model, volume, patch_size, device, batch_size=4):
    """Single-pass sliding window inference on a full 3D volume."""
    model.eval()
    C_in = volume.shape[0]
    _, D, H, W = volume.shape
    stride = patch_size // 2

    pred_sum = torch.zeros(OUT_CHANNELS, D, H, W, device='cpu')
    count = torch.zeros(OUT_CHANNELS, D, H, W, device='cpu')

    coords = []
    for d0 in range(0, max(1, D - patch_size + 1), stride):
        for h0 in range(0, max(1, H - patch_size + 1), stride):
            for w0 in range(0, max(1, W - patch_size + 1), stride):
                d0 = min(d0, max(0, D - patch_size))
                h0 = min(h0, max(0, H - patch_size))
                w0 = min(w0, max(0, W - patch_size))
                coords.append((d0, h0, w0))
    coords = list(set(coords))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            patches = []
            for (d0, h0, w0) in batch_coords:
                patch = volume[:, d0:d0+patch_size, h0:h0+patch_size, w0:w0+patch_size]
                _, pd, ph, pw = patch.shape
                if pd < patch_size or ph < patch_size or pw < patch_size:
                    padded = torch.zeros(C_in, patch_size, patch_size, patch_size)
                    padded[:, :pd, :ph, :pw] = patch
                    patch = padded
                patches.append(patch)
            batch_tensor = torch.stack(patches).to(device)
            out = torch.sigmoid(model(batch_tensor)).cpu()
            for j, (d0, h0, w0) in enumerate(batch_coords):
                d_end = min(d0 + patch_size, D)
                h_end = min(h0 + patch_size, H)
                w_end = min(w0 + patch_size, W)
                pred_sum[:, d0:d_end, h0:h_end, w0:w_end] += out[j, :, :d_end-d0, :h_end-h0, :w_end-w0]
                count[:, d0:d_end, h0:h_end, w0:w_end] += 1

    count = torch.clamp(count, min=1)
    return pred_sum / count  # (C, D, H, W)


def sliding_window_inference(model, volume, patch_size, device, batch_size=4, tta=True):
    """Sliding window with test-time augmentation (8 flips)."""
    pred = _sliding_window_single(model, volume, patch_size, device, batch_size)
    if not tta:
        return pred

    # TTA: flip along each spatial axis and combinations
    # volume is (C, D, H, W) = dims 0,1,2,3
    flip_dims = [
        (1,), (2,), (3,),       # single flips
        (1, 2), (1, 3), (2, 3), # double flips
        (1, 2, 3),              # triple flip
    ]
    n_aug = 1
    for dims in flip_dims:
        vol_flipped = torch.flip(volume, dims=list(dims))
        pred_flipped = _sliding_window_single(model, vol_flipped, patch_size, device, batch_size)
        # Flip prediction back
        pred += torch.flip(pred_flipped, dims=list(dims))
        n_aug += 1

    return pred / n_aug


def load_model(model_path, device):
    model = create_model_3d("unet3d", in_channels=4, out_channels=OUT_CHANNELS, features=[16, 32, 64, 128])
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset to get cases
    val_dataset = BraTSPatchDataset(
        root=DATA_ROOT, split="val", patch_size=PATCH_SIZE,
        target=TARGET, use_preprocessed=True,
    )

    # Pick N_CASES with decent tumor size for visualization
    rng = np.random.RandomState(42)
    case_indices = list(range(len(val_dataset.cases)))
    rng.shuffle(case_indices)

    # Load one model to help select best slices (use best-performing method)
    ref_model_name = list(METHODS.keys())[-1]  # Shampoo (oracle)
    ref_model = load_model(METHODS[ref_model_name], device)
    print(f"Using {ref_model_name} to select best slices...")

    # Find cases with good tumor presence AND high Dice on best slice
    candidates = []
    for idx in case_indices:
        case_path = val_dataset.cases[idx]
        image, seg = _load_volume_npz(case_path)
        mask = _binarize_target(seg, TARGET)
        # mask shape: (C, D, H, W) for multilabel
        if mask.ndim == 3:
            tumor_slices = mask.sum(axis=(1, 2))
        else:
            wt_idx = CHANNEL_NAMES.index("WT")
            tumor_slices = mask[wt_idx].sum(axis=(1, 2))

        # Only consider cases with decent tumor
        max_area = tumor_slices.max()
        if max_area < 200:
            continue

        # Run inference to find slice with best Dice
        volume_t = torch.from_numpy(image.copy()).float()
        pred_vol = sliding_window_inference(ref_model, volume_t, PATCH_SIZE, device)

        # Compute per-slice mean Dice across channels
        n_slices = mask.shape[1] if mask.ndim == 4 else mask.shape[0]
        slice_dices = []
        for s in range(n_slices):
            if tumor_slices[s] < 50:  # skip slices with tiny tumor
                slice_dices.append(0.0)
                continue
            dices = []
            for c in range(len(CHANNEL_NAMES)):
                gt_s = mask[c, s] if mask.ndim == 4 else mask[s]
                pred_s = (pred_vol[c, s].numpy() > THRESHOLD).astype(float)
                intersection = (pred_s * gt_s).sum()
                union = pred_s.sum() + gt_s.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                dices.append(dice)
            slice_dices.append(np.mean(dices))

        best_slice = int(np.argmax(slice_dices))
        best_dice = slice_dices[best_slice]
        candidates.append((best_dice, idx, case_path, image, mask, best_slice))

        if len(candidates) >= N_CASES * 3:  # collect enough to pick top N
            break

    # Sort by Dice descending, pick top N_CASES
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [(c[1], c[2], c[3], c[4], c[5]) for c in candidates[:N_CASES]]
    print(f"Selected {len(selected)} cases (best slice Dice: "
          f"{', '.join(f'{c[0]:.2f}' for c in candidates[:N_CASES])})")

    print(f"Selected {len(selected)} cases for visualization")

    # Load all models
    models = {}
    for name, path in METHODS.items():
        print(f"Loading {name} from {path}...")
        models[name] = load_model(path, device)

    # Generate predictions
    method_names = list(METHODS.keys())
    n_cols = 2 + len(method_names)  # Image + GT + methods

    fig, axes = plt.subplots(N_CASES, n_cols, figsize=(2.5 * n_cols, 2.5 * N_CASES))
    if N_CASES == 1:
        axes = axes[np.newaxis, :]

    # Same colors for GT and prediction per class
    class_colors = {'ET': 'yellow', 'TC': 'cyan', 'WT': 'lime'}
    # GT = solid contour, Pred = dashed contour (same color)
    CROP_PAD = 15  # pixels padding around tumor bbox

    # First pass: compute crop regions and predictions
    row_data = []
    for row, (idx, case_path, image, mask, best_slice) in enumerate(selected):
        print(f"Case {row+1}/{N_CASES}: {Path(case_path).stem}, slice {best_slice}")

        img_slice = image[0, best_slice]  # (H, W)
        gt_slices = {}
        for c, name in enumerate(CHANNEL_NAMES):
            gt_slices[name] = mask[c, best_slice]

        # Compute crop bbox from WT (largest region)
        wt_mask = gt_slices["WT"]
        ys, xs = np.where(wt_mask > 0)
        y0 = max(0, ys.min() - CROP_PAD)
        y1 = min(img_slice.shape[0], ys.max() + CROP_PAD)
        x0 = max(0, xs.min() - CROP_PAD)
        x1 = min(img_slice.shape[1], xs.max() + CROP_PAD)

        # Run predictions
        volume_t = torch.from_numpy(image.copy()).float()
        preds = {}
        preds_probs = {}
        for method_name in method_names:
            pred_vol = sliding_window_inference(
                models[method_name], volume_t, PATCH_SIZE, device
            )
            pred_slices = {}
            prob_slices = {}
            dices = []
            for c, name in enumerate(CHANNEL_NAMES):
                prob_s = pred_vol[c, best_slice].numpy()
                pred_s = (prob_s > THRESHOLD).astype(float)
                pred_slices[name] = pred_s
                prob_slices[name] = prob_s
                gt_s = gt_slices[name]
                intersection = (pred_s * gt_s).sum()
                union = pred_s.sum() + gt_s.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                dices.append(dice)
            preds[method_name] = (pred_slices, np.mean(dices))
            preds_probs[method_name] = prob_slices

        row_data.append((img_slice, gt_slices, preds, preds_probs, (y0, y1, x0, x1)))

    # Create figure with no spacing
    fig, axes = plt.subplots(N_CASES, n_cols, figsize=(2.2 * n_cols, 2.2 * N_CASES))
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    if N_CASES == 1:
        axes = axes[np.newaxis, :]

    for row, (img_slice, gt_slices, preds, preds_probs, (y0, y1, x0, x1)) in enumerate(row_data):
        # Crop all slices
        img_crop = img_slice[y0:y1, x0:x1]

        # Show image
        axes[row, 0].imshow(img_crop, cmap='gray', interpolation='nearest')
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title("FLAIR", fontsize=11, fontweight='bold')

        # Show ground truth
        axes[row, 1].imshow(img_crop, cmap='gray', interpolation='nearest')
        for name in CHANNEL_NAMES:
            gt_crop = gt_slices[name][y0:y1, x0:x1]
            if gt_crop.sum() > 0:
                axes[row, 1].contour(gt_crop, levels=[0.5],
                                     colors=class_colors[name], linewidths=1.5)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title("Ground Truth", fontsize=11, fontweight='bold')

        # Show each method
        for col, method_name in enumerate(method_names):
            pred_slices_dict, mean_dice = preds[method_name]
            pred_probs_dict = preds_probs[method_name]
            ax = axes[row, col + 2]
            ax.imshow(img_crop, cmap='gray', interpolation='nearest')

            # Build filled overlay: argmax over channels where pred > threshold
            h_crop, w_crop = img_crop.shape
            overlay = np.zeros((h_crop, w_crop, 4), dtype=np.float32)
            # Stack probabilities for argmax
            prob_stack = np.stack([pred_probs_dict[name][y0:y1, x0:x1]
                                   for name in CHANNEL_NAMES], axis=0)  # (C, H, W)
            binary_stack = (prob_stack > THRESHOLD)
            any_pred = binary_stack.any(axis=0)  # (H, W)
            winner = np.argmax(prob_stack, axis=0)  # (H, W)

            for c, name in enumerate(CHANNEL_NAMES):
                rgb = np.array(mcolors.to_rgb(class_colors[name]))
                mask_c = any_pred & (winner == c)
                overlay[mask_c, :3] = rgb
                overlay[mask_c, 3] = 0.45

            ax.imshow(overlay, interpolation='nearest')

            # GT contours on top
            for name in CHANNEL_NAMES:
                gt_crop = gt_slices[name][y0:y1, x0:x1]
                if gt_crop.sum() > 0:
                    ax.contour(gt_crop, levels=[0.5],
                               colors=class_colors[name], linewidths=1.2)

            ax.axis('off')
            ax.text(
                0.97, 0.03, f"{mean_dice:.2f}",
                transform=ax.transAxes,
                fontsize=9, color='white', fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
            )
            if row == 0:
                ax.set_title(method_name, fontsize=11, fontweight='bold')

    # Legend: contour = GT, filled = Pred, one per class
    legend_handles = []
    for name in CHANNEL_NAMES:
        legend_handles.append(Line2D([0], [0], color=class_colors[name], linewidth=1.5,
                                     linestyle='solid', label=f'{name} (GT)'))
        legend_handles.append(mpatches.Patch(facecolor=class_colors[name], alpha=0.45,
                                             label=f'{name} (Pred)'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(CHANNEL_NAMES) * 2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))

    out_path = "figures/brats_qualitative.pdf"
    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out_path} and .png")


if __name__ == "__main__":
    main()
