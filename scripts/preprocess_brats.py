#!/usr/bin/env python3
"""
Preprocess BraTS NIfTI files to fast NumPy format.

Converts each volume from 4 NIfTI files + 1 seg to a single .npz file
containing pre-normalized image (4, D, H, W) and segmentation (D, H, W).

This speeds up training by 10-20x by avoiding NIfTI decompression overhead.

Usage:
    python scripts/preprocess_brats.py
    python scripts/preprocess_brats.py --output_dir /tmp/brats_npz --num_workers 8
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import nibabel as nib
import numpy as np


BRATS_DEFAULT_ROOT = "/eos/project/d/diagbox/BRATS2025/Glioma_seg_pre_post_treatment_mri/PRE/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS2025-GLI-PRE-Challenge-TrainingData"
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


def process_case(case_path: Path, output_dir: Path) -> str:
    """Process a single BraTS case to .npz format."""
    case_name = case_path.name
    output_path = output_dir / f"{case_name}.npz"
    
    # Skip if already processed
    if output_path.exists():
        return f"Skipped {case_name} (already exists)"
    
    try:
        # Load all modalities
        channels = []
        for mod in MODALITIES:
            nii_path = case_path / f"{case_name}-{mod}.nii.gz"
            vol = nib.load(str(nii_path)).get_fdata().astype(np.float32)
            
            # Per-modality z-score normalization (brain mask only)
            brain_mask = vol > 0
            if brain_mask.sum() > 0:
                mean = vol[brain_mask].mean()
                std = vol[brain_mask].std()
                vol = np.where(brain_mask, (vol - mean) / (std + 1e-8), 0.0)
            
            channels.append(vol)
        
        image = np.stack(channels, axis=0).astype(np.float16)  # (4, D, H, W), float16 to save space
        
        # Load segmentation
        seg_path = case_path / f"{case_name}-seg.nii.gz"
        seg = nib.load(str(seg_path)).get_fdata().astype(np.uint8)  # Labels are 0-3
        
        # Save as compressed npz
        np.savez_compressed(output_path, image=image, seg=seg)
        
        return f"Processed {case_name}"
    
    except Exception as e:
        return f"Error {case_name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS to NumPy format")
    parser.add_argument("--input_dir", type=str, default=BRATS_DEFAULT_ROOT,
                        help="Path to BraTS training data")
    parser.add_argument("--output_dir", type=str, 
                        default="/eos/project/d/diagbox/BRATS2025/preprocessed_npz",
                        help="Output directory for .npz files")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all cases
    cases = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases in {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_case, case, output_dir): case 
                   for case in cases}
        
        with tqdm(total=len(cases), desc="Preprocessing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.set_postfix_str(result[:50])
                pbar.update(1)
    
    # Summary
    npz_files = list(output_dir.glob("*.npz"))
    total_size_gb = sum(f.stat().st_size for f in npz_files) / (1024**3)
    print(f"\nDone! {len(npz_files)} files, {total_size_gb:.1f} GB total")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
