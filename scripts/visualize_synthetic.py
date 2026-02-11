#!/usr/bin/env python3
"""
Visualize all synthetic data generators (2D as PNG, 3D as NIfTI + mid-slice PNG).

Foreground regions are visually distinct (intensity shift + edge highlighting),
making synthetic data look like real segmentation tasks.

Saves to figures/synthetic/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

from src.data.synthetic import (
    get_noise_generator,
    get_mask_generator,
    composite_image_with_mask,
)
from src.data.synthetic3d import (
    get_noise_generator_3d,
    get_mask_generator_3d,
    composite_image_with_mask_3d,
)

OUT = Path("figures/synthetic")
OUT.mkdir(parents=True, exist_ok=True)

NOISE_TYPES = ["white", "pink", "brown"]
MASK_STRATEGIES = ["gaussian_blobs", "random_shapes", "pink_threshold", "voronoi", "frangi"]

SIZE_2D = 256
SIZE_3D = 64

np.random.seed(42)


def make_composite_2d(noise_fn, mask_fn, size, seed=42):
    """Generate a composited 2D image+mask pair."""
    np.random.seed(seed)
    bg = noise_fn(size)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    mask = mask_fn(size)
    shift = np.random.uniform(0.2, 0.4) * np.random.choice([-1, 1])
    img = composite_image_with_mask(
        bg, mask, fg_intensity_shift=shift,
        fg_noise_scale=0.1, edge_width=max(1, size // 64), edge_brightness=0.15,
    )
    return bg, img, mask


def make_composite_3d(noise_fn, mask_fn, size, seed=42):
    """Generate a composited 3D image+mask pair."""
    np.random.seed(seed)
    bg = noise_fn(size)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    mask = mask_fn(size)
    shift = np.random.uniform(0.2, 0.4) * np.random.choice([-1, 1])
    img = composite_image_with_mask_3d(
        bg, mask, fg_intensity_shift=shift,
        fg_noise_scale=0.1, edge_width=max(1, size // 32), edge_brightness=0.15,
    )
    return bg, img, mask


# =========================================================================
# 2D
# =========================================================================
print("=== 2D Synthetic Data ===")

# -- Before/After comparison for one example --
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
np.random.seed(42)
nf = get_noise_generator("pink")
mf = get_mask_generator("gaussian_blobs")
bg, img, mask = make_composite_2d(nf, mf, SIZE_2D)

axes[0, 0].imshow(bg, cmap="gray"); axes[0, 0].set_title("Background noise", fontsize=12)
axes[0, 1].imshow(mask, cmap="gray"); axes[0, 1].set_title("Mask", fontsize=12)
axes[0, 2].imshow(bg, cmap="gray")
axes[0, 2].contour(mask, levels=[0.5], colors="red", linewidths=1.5)
axes[0, 2].set_title("Before (no compositing)", fontsize=12)

axes[1, 0].imshow(img, cmap="gray"); axes[1, 0].set_title("Composited image", fontsize=12)
axes[1, 1].imshow(mask, cmap="gray"); axes[1, 1].set_title("Mask", fontsize=12)
axes[1, 2].imshow(img, cmap="gray")
axes[1, 2].contour(mask, levels=[0.5], colors="red", linewidths=1.5)
axes[1, 2].set_title("After (with compositing)", fontsize=12)

for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Realistic Compositing: Before vs After", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "compositing_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved compositing_before_after.png")

# -- Combined grid: noise x mask (with compositing) --
fig, axes = plt.subplots(
    len(NOISE_TYPES), len(MASK_STRATEGIES),
    figsize=(4 * len(MASK_STRATEGIES), 4 * len(NOISE_TYPES)),
)
for i, nt in enumerate(NOISE_TYPES):
    for j, ms in enumerate(MASK_STRATEGIES):
        seed = 42 + i * 100 + j
        nf = get_noise_generator(nt)
        mf = get_mask_generator(ms)
        _, img, mask = make_composite_2d(nf, mf, SIZE_2D, seed=seed)

        ax = axes[i, j]
        ax.imshow(img, cmap="gray")
        ax.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
        if i == 0:
            ax.set_title(ms.replace("_", "\n"), fontsize=11)
        if j == 0:
            ax.set_ylabel(nt, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("2D Synthetic Data: Noise x Mask (composited)", fontsize=18, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "2d_grid_noise_x_mask.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 2d_grid_noise_x_mask.png")

# -- Noise gallery --
fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(4 * len(NOISE_TYPES), 4))
for ax, nt in zip(axes, NOISE_TYPES):
    np.random.seed(42)
    noise = get_noise_generator(nt)(SIZE_2D)
    ax.imshow(noise, cmap="gray")
    ax.set_title(nt, fontsize=14); ax.axis("off")
fig.suptitle("2D Noise Types", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "2d_noise_types.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 2d_noise_types.png")

# -- Mask gallery --
fig, axes = plt.subplots(1, len(MASK_STRATEGIES), figsize=(4 * len(MASK_STRATEGIES), 4))
for ax, ms in zip(axes, MASK_STRATEGIES):
    np.random.seed(42)
    mask = get_mask_generator(ms)(SIZE_2D)
    ax.imshow(mask, cmap="gray")
    ax.set_title(ms.replace("_", " "), fontsize=12); ax.axis("off")
fig.suptitle("2D Mask Strategies", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "2d_mask_strategies.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 2d_mask_strategies.png")

# -- Individual pairs --
for nt in NOISE_TYPES:
    for ms in MASK_STRATEGIES:
        nf = get_noise_generator(nt)
        mf = get_mask_generator(ms)
        _, img, mask = make_composite_2d(nf, mf, SIZE_2D, seed=42)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(img, cmap="gray"); ax1.set_title(f"Image ({nt})"); ax1.axis("off")
        ax2.imshow(mask, cmap="gray"); ax2.set_title(f"Mask ({ms})"); ax2.axis("off")
        ax3.imshow(img, cmap="gray")
        ax3.contour(mask, levels=[0.5], colors="red", linewidths=1.5)
        ax3.set_title("Overlay"); ax3.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"2d_{nt}_{ms}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

print(f"  Saved {len(NOISE_TYPES) * len(MASK_STRATEGIES)} individual 2D pair PNGs")


# =========================================================================
# 3D
# =========================================================================
print("\n=== 3D Synthetic Data ===")

# -- 3D Before/After comparison --
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
nf3 = get_noise_generator_3d("pink")
mf3 = get_mask_generator_3d("gaussian_blobs")
bg3, img3, mask3 = make_composite_3d(nf3, mf3, SIZE_3D)
mid = SIZE_3D // 2

axes[0, 0].imshow(bg3[mid], cmap="gray"); axes[0, 0].set_title("Background noise")
axes[0, 1].imshow(mask3[mid], cmap="gray"); axes[0, 1].set_title("Mask")
axes[0, 2].imshow(bg3[mid], cmap="gray")
axes[0, 2].contour(mask3[mid], levels=[0.5], colors="red", linewidths=1.5)
axes[0, 2].set_title("Before (no compositing)")

axes[1, 0].imshow(img3[mid], cmap="gray"); axes[1, 0].set_title("Composited image")
axes[1, 1].imshow(mask3[mid], cmap="gray"); axes[1, 1].set_title("Mask")
axes[1, 2].imshow(img3[mid], cmap="gray")
axes[1, 2].contour(mask3[mid], levels=[0.5], colors="red", linewidths=1.5)
axes[1, 2].set_title("After (with compositing)")

for ax in axes.flat:
    ax.axis("off")
fig.suptitle("3D Compositing: Before vs After (axial mid-slice)", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "3d_compositing_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 3d_compositing_before_after.png")

# -- Noise gallery (mid-slices) --
fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(4 * len(NOISE_TYPES), 4))
for ax, nt in zip(axes, NOISE_TYPES):
    np.random.seed(42)
    noise = get_noise_generator_3d(nt)(SIZE_3D)
    ax.imshow(noise[mid], cmap="gray")
    ax.set_title(nt, fontsize=14); ax.axis("off")
fig.suptitle(f"3D Noise Types (axial mid-slice, {SIZE_3D}^3)", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "3d_noise_types.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 3d_noise_types.png")

# -- Mask gallery (mid-slices) --
fig, axes = plt.subplots(1, len(MASK_STRATEGIES), figsize=(4 * len(MASK_STRATEGIES), 4))
for ax, ms in zip(axes, MASK_STRATEGIES):
    np.random.seed(42)
    mask = get_mask_generator_3d(ms)(SIZE_3D)
    ax.imshow(mask[mid], cmap="gray")
    ax.set_title(ms.replace("_", " "), fontsize=12); ax.axis("off")
fig.suptitle(f"3D Mask Strategies (axial mid-slice, {SIZE_3D}^3)", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "3d_mask_strategies.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 3d_mask_strategies.png")

# -- Combined grid 3D --
fig, axes = plt.subplots(
    len(NOISE_TYPES), len(MASK_STRATEGIES),
    figsize=(4 * len(MASK_STRATEGIES), 4 * len(NOISE_TYPES)),
)
for i, nt in enumerate(NOISE_TYPES):
    for j, ms in enumerate(MASK_STRATEGIES):
        seed = 42 + i * 100 + j
        nf = get_noise_generator_3d(nt)
        mf = get_mask_generator_3d(ms)
        _, img, mask = make_composite_3d(nf, mf, SIZE_3D, seed=seed)

        ax = axes[i, j]
        ax.imshow(img[mid], cmap="gray")
        ax.contour(mask[mid], levels=[0.5], colors="red", linewidths=1.5)
        if i == 0:
            ax.set_title(ms.replace("_", "\n"), fontsize=11)
        if j == 0:
            ax.set_ylabel(nt, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"3D Synthetic Data: Noise x Mask (composited, axial mid-slice, {SIZE_3D}^3)",
             fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "3d_grid_noise_x_mask.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 3d_grid_noise_x_mask.png")

# -- NIfTI volumes (composited) --
affine = np.eye(4)

for nt in NOISE_TYPES:
    np.random.seed(42)
    noise = get_noise_generator_3d(nt)(SIZE_3D)
    noise = ((noise - noise.min()) / (noise.max() - noise.min() + 1e-8)).astype(np.float32)
    nib.save(nib.Nifti1Image(noise, affine), str(OUT / f"3d_noise_{nt}.nii.gz"))

for ms in MASK_STRATEGIES:
    np.random.seed(42)
    mask = get_mask_generator_3d(ms)(SIZE_3D)
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), str(OUT / f"3d_mask_{ms}.nii.gz"))

# Composited 4-channel + mask NIfTI pairs
for nt in ["pink", "brown"]:
    for ms in ["gaussian_blobs", "frangi", "voronoi"]:
        np.random.seed(42)
        nf = get_noise_generator_3d(nt)
        mf = get_mask_generator_3d(ms)
        mask = mf(SIZE_3D)

        channels = []
        for c in range(4):
            ch = nf(SIZE_3D)
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            shift = np.random.uniform(0.2, 0.4) * np.random.choice([-1, 1])
            ch = composite_image_with_mask_3d(
                ch, mask, fg_intensity_shift=shift,
                fg_noise_scale=0.1, edge_width=max(1, SIZE_3D // 32), edge_brightness=0.15,
            )
            channels.append(ch)
        image = np.stack(channels, axis=-1).astype(np.float32)

        nib.save(nib.Nifti1Image(image, affine), str(OUT / f"3d_sample_{nt}_{ms}_image.nii.gz"))
        nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), str(OUT / f"3d_sample_{nt}_{ms}_mask.nii.gz"))

print(f"  Saved {len(NOISE_TYPES)} noise + {len(MASK_STRATEGIES)} mask + 6 composited sample NIfTIs")

# -- 3-view montage for selected composited pairs --
for nt, ms in [("pink", "gaussian_blobs"), ("brown", "frangi"), ("white", "voronoi")]:
    nf = get_noise_generator_3d(nt)
    mf = get_mask_generator_3d(ms)
    _, img, mask = make_composite_3d(nf, mf, SIZE_3D)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    views = [
        (img[mid, :, :], mask[mid, :, :], "Axial"),
        (img[:, mid, :], mask[:, mid, :], "Coronal"),
        (img[:, :, mid], mask[:, :, mid], "Sagittal"),
    ]
    for col, (img_s, msk_s, vname) in enumerate(views):
        axes[0, col].imshow(img_s, cmap="gray")
        axes[0, col].set_title(f"{vname} - Image", fontsize=12); axes[0, col].axis("off")
        axes[1, col].imshow(msk_s, cmap="gray")
        axes[1, col].contour(msk_s, levels=[0.5], colors="red", linewidths=1)
        axes[1, col].set_title(f"{vname} - Mask", fontsize=12); axes[1, col].axis("off")

    fig.suptitle(f"3D {nt} + {ms} (composited, {SIZE_3D}^3)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"3d_3view_{nt}_{ms}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print("  Saved 3 three-view montage PNGs")

print(f"\nAll figures saved to {OUT}/")
