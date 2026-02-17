#!/usr/bin/env python3
"""Preprocess Hepatic Vessel CT volumes for efficient training.

Steps:
    1. Resample to isotropic spacing (default 1.5mm³)
    2. CT windowing [-200, 300] HU + z-score normalization
    3. Crop to foreground bounding box + padding
    4. Save as .npz for fast loading

Usage:
    python scripts/preprocess_hepatic.py \
        --input_dir data/Task08_HepaticVessel \
        --output_dir data/hepatic_preprocessed \
        --target_spacing 1.5
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def resample_volume(vol, original_spacing, target_spacing, order=3):
    """Resample volume to target isotropic spacing."""
    scale = np.array(original_spacing) / target_spacing
    new_shape = np.round(np.array(vol.shape) * scale).astype(int)
    resampled = zoom(vol, scale, order=order)
    return resampled


def preprocess_case(img_path, label_path, output_dir, target_spacing, padding):
    """Preprocess a single case: resample, crop, normalize, save."""
    case_name = img_path.stem.replace(".nii", "")

    # Load
    img_nib = nib.load(str(img_path))
    original_spacing = np.array(img_nib.header.get_zooms()[:3], dtype=np.float64)
    vol = img_nib.get_fdata().astype(np.float32)
    seg = nib.load(str(label_path)).get_fdata().astype(np.int64)

    # Resample to isotropic
    scale = original_spacing / target_spacing
    vol_resampled = zoom(vol, scale, order=3)
    seg_resampled = zoom(seg.astype(np.float32), scale, order=0).astype(np.int64)

    # CT windowing + z-score
    vol_resampled = np.clip(vol_resampled, -200, 300)
    mean, std = vol_resampled.mean(), vol_resampled.std()
    vol_resampled = (vol_resampled - mean) / (std + 1e-8)

    # Crop to foreground bounding box
    fg_coords = np.argwhere(seg_resampled > 0)
    if len(fg_coords) == 0:
        # No foreground — save full volume (rare edge case)
        crop_min = np.array([0, 0, 0])
        crop_max = np.array(vol_resampled.shape) - 1
    else:
        crop_min = fg_coords.min(axis=0) - padding
        crop_max = fg_coords.max(axis=0) + padding
        crop_min = np.maximum(crop_min, 0)
        crop_max = np.minimum(crop_max, np.array(vol_resampled.shape) - 1)

    vol_crop = vol_resampled[
        crop_min[0]:crop_max[0]+1,
        crop_min[1]:crop_max[1]+1,
        crop_min[2]:crop_max[2]+1,
    ]
    seg_crop = seg_resampled[
        crop_min[0]:crop_max[0]+1,
        crop_min[1]:crop_max[1]+1,
        crop_min[2]:crop_max[2]+1,
    ]

    # Save
    out_path = output_dir / f"{case_name}.npz"
    np.savez_compressed(
        out_path,
        image=vol_crop.astype(np.float32),
        label=seg_crop.astype(np.uint8),
        original_spacing=original_spacing,
        target_spacing=np.array([target_spacing]*3),
        crop_min=crop_min,
        crop_max=crop_max,
        original_shape=np.array(vol.shape),
        resampled_shape=np.array(vol_resampled.shape),
    )

    fg_pct = (seg_crop > 0).sum() / seg_crop.size * 100
    return case_name, vol_crop.shape, fg_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/Task08_HepaticVessel")
    parser.add_argument("--output_dir", type=str, default="data/hepatic_preprocessed")
    parser.add_argument("--target_spacing", type=float, default=1.5,
                        help="Isotropic spacing in mm (default 1.5)")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding around foreground bbox in voxels")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover cases
    images_dir = input_dir / "imagesTr"
    labels_dir = input_dir / "labelsTr"

    cases = []
    for img_path in sorted(images_dir.glob("hepaticvessel_*.nii.gz")):
        label_path = labels_dir / img_path.name
        if label_path.exists():
            cases.append((img_path, label_path))

    print(f"Found {len(cases)} cases")
    print(f"Target spacing: {args.target_spacing}mm isotropic")
    print(f"Padding: {args.padding} voxels")
    print(f"Output: {output_dir}")

    def _process(case):
        img_path, label_path = case
        return preprocess_case(img_path, label_path, output_dir,
                               args.target_spacing, args.padding)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(_process, cases), total=len(cases), desc="Preprocessing"):
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Preprocessed {len(results)} cases to {output_dir}")
    shapes = [r[1] for r in results]
    fg_pcts = [r[2] for r in results]
    print(f"  Cropped shapes (min): {tuple(np.min(shapes, axis=0))}")
    print(f"  Cropped shapes (max): {tuple(np.max(shapes, axis=0))}")
    print(f"  Cropped shapes (median): {tuple(np.median(shapes, axis=0).astype(int))}")
    print(f"  FG percentage: {np.mean(fg_pcts):.2f}% (min={np.min(fg_pcts):.2f}%, max={np.max(fg_pcts):.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
