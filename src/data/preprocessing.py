"""
Preprocessing utilities for retinal vessel segmentation datasets.

Supported datasets:
- DRIVE: 40 images (20 train, 20 test) at 565x584 resolution
- STARE: 397 images total, 20 with vessel segmentation labels at 700x605 resolution
- HRF: 45 high-resolution images (3504x2336) with vessel and FOV annotations

For differential privacy experiments with AdaDPS:
- Private data: Small labeled dataset for DP-SGD training
- Public data: Larger dataset (labeled or unlabeled) for preconditioner estimation
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def load_ppm_image(path: str) -> np.ndarray:
    """Load a PPM image file."""
    return np.array(Image.open(path))


def load_tif_image(path: str) -> np.ndarray:
    """Load a TIFF image file."""
    return np.array(Image.open(path))


def load_gif_mask(path: str) -> np.ndarray:
    """Load a GIF mask file and convert to binary."""
    img = Image.open(path)
    mask = np.array(img)
    # Convert to binary (0 or 1)
    return (mask > 0).astype(np.uint8)


def create_fov_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Create field-of-view mask from fundus image.
    The FOV is the circular region containing the actual retinal image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Threshold to find FOV
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return (mask > 0).astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    return image.astype(np.float32) / 255.0


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Extract green channel from RGB image.
    Green channel typically has best vessel contrast in fundus images.
    """
    if len(image.shape) == 3:
        return image[:, :, 1]
    return image


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast while limiting noise amplification.
    """
    if len(image.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def resize_and_pad(image: np.ndarray, target_size: Tuple[int, int],
                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Resize image to target size while maintaining aspect ratio with padding.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)

    # Center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    if mask is not None:
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_padded = np.zeros((target_h, target_w), dtype=mask.dtype)
        mask_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = mask_resized
        return padded, mask_padded

    return padded, None


def preprocess_drive(
    data_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe: bool = True,
    use_green_channel: bool = False,
) -> None:
    """
    Preprocess DRIVE dataset for training.

    DRIVE dataset structure:
    - training/images/: XX_training.tif (20 images)
    - training/1st_manual/: XX_manual1.gif (vessel annotations)
    - training/mask/: XX_training_mask.gif (FOV masks)
    - test/images/: XX_test.tif (20 images)
    - test/mask/: XX_test_mask.gif (FOV masks)
    - test/1st_manual/ or test/2nd_manual/: ground truth (if available)

    Args:
        data_dir: Path to DRIVE dataset root
        output_dir: Path to save preprocessed data
        target_size: Target image size (height, width)
        apply_clahe: Whether to apply CLAHE enhancement
        use_green_channel: Whether to use only green channel
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    for split in ["training", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            print(f"Warning: {split} split not found")
            continue

        # Create output directories
        out_images = output_path / split / "images"
        out_masks = output_path / split / "masks"
        out_fov = output_path / split / "fov"

        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        out_fov.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_dir = split_path / "images"
        image_files = sorted(image_dir.glob("*.tif"))

        print(f"Processing DRIVE {split} set ({len(image_files)} images)...")

        for img_path in tqdm(image_files):
            # Extract image ID (e.g., "21" from "21_training.tif")
            img_id = img_path.stem.split("_")[0]

            # Load image
            image = load_tif_image(str(img_path))

            # Load FOV mask
            suffix = "training" if split == "training" else "test"
            fov_path = split_path / "mask" / f"{img_id}_{suffix}_mask.gif"
            if fov_path.exists():
                fov_mask = load_gif_mask(str(fov_path))
            else:
                fov_mask = create_fov_mask(image)

            # Load vessel annotation (1st_manual) if available
            manual_path = split_path / "1st_manual" / f"{img_id}_manual1.gif"
            if manual_path.exists():
                vessel_mask = load_gif_mask(str(manual_path))
            else:
                vessel_mask = None

            # Apply preprocessing
            if apply_clahe:
                image = clahe_enhancement(image)

            if use_green_channel:
                image = extract_green_channel(image)

            # Resize to target size
            image, fov_mask = resize_and_pad(image, target_size, fov_mask)
            if vessel_mask is not None:
                vessel_mask, _ = resize_and_pad(vessel_mask, target_size)

            # Save preprocessed data
            Image.fromarray(image).save(out_images / f"{img_id}.png")
            Image.fromarray((fov_mask * 255).astype(np.uint8)).save(out_fov / f"{img_id}.png")
            if vessel_mask is not None:
                Image.fromarray((vessel_mask * 255).astype(np.uint8)).save(out_masks / f"{img_id}.png")

    print(f"DRIVE preprocessing complete. Output saved to {output_dir}")


def preprocess_stare(
    data_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe: bool = True,
    use_green_channel: bool = False,
    annotator: str = "ah",  # "ah" (Adam Hoover) or "vk" (Valentina Kouznetsova)
    labeled_only: bool = True,  # If False, process all 397 images
) -> None:
    """
    Preprocess STARE dataset for training.

    STARE dataset structure:
    - archive/: imXXXX.ppm (397 total images)
    - images/: imXXXX.ppm (20 images with labels - subset of archive)
    - labels-ah/: imXXXX.ah.ppm (annotations by Adam Hoover)
    - labels-vk/: imXXXX.vk.ppm (annotations by Valentina Kouznetsova)

    Args:
        data_dir: Path to STARE dataset root
        output_dir: Path to save preprocessed data
        target_size: Target image size (height, width)
        apply_clahe: Whether to apply CLAHE enhancement
        use_green_channel: Whether to use only green channel
        annotator: Which annotator's labels to use ("ah" or "vk")
        labeled_only: If True, only process 20 labeled images; if False, process all 397
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Create output directories
    out_images = output_path / "images"
    out_masks = output_path / "masks"
    out_fov = output_path / "fov"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_fov.mkdir(parents=True, exist_ok=True)

    # Get image files - either from labeled subset or full archive
    if labeled_only:
        image_dir = data_path / "images"
        image_files = sorted(image_dir.glob("*.ppm"))
        if not image_files:
            # Fallback: get images that have labels
            label_dir = data_path / f"labels-{annotator}"
            labeled_ids = {p.stem.replace(f".{annotator}", "") for p in label_dir.glob("*.ppm")}
            archive_dir = data_path / "archive"
            image_files = [archive_dir / f"{id_}.ppm" for id_ in sorted(labeled_ids)
                          if (archive_dir / f"{id_}.ppm").exists()]
    else:
        # Process all images from archive
        image_dir = data_path / "archive"
        image_files = sorted(image_dir.glob("*.ppm"))

    print(f"Processing STARE dataset ({len(image_files)} images, labeled_only={labeled_only})...")

    for img_path in tqdm(image_files):
        img_id = img_path.stem  # e.g., "im0001"

        # Load image
        image = load_ppm_image(str(img_path))

        # Load vessel annotation if available
        label_path = data_path / f"labels-{annotator}" / f"{img_id}.{annotator}.ppm"
        if label_path.exists():
            vessel_mask = load_ppm_image(str(label_path))
            # Convert to binary (annotations are white on black)
            if len(vessel_mask.shape) == 3:
                vessel_mask = vessel_mask[:, :, 0]
            vessel_mask = (vessel_mask > 0).astype(np.uint8)
        else:
            vessel_mask = None

        # Create FOV mask (STARE doesn't provide FOV masks)
        fov_mask = create_fov_mask(image)

        # Apply preprocessing
        if apply_clahe:
            image = clahe_enhancement(image)

        if use_green_channel:
            image = extract_green_channel(image)

        # Resize to target size
        image, fov_mask = resize_and_pad(image, target_size, fov_mask)
        if vessel_mask is not None:
            vessel_mask, _ = resize_and_pad(vessel_mask, target_size)

        # Save preprocessed data
        Image.fromarray(image).save(out_images / f"{img_id}.png")
        Image.fromarray((fov_mask * 255).astype(np.uint8)).save(out_fov / f"{img_id}.png")
        if vessel_mask is not None:
            Image.fromarray((vessel_mask * 255).astype(np.uint8)).save(out_masks / f"{img_id}.png")

    print(f"STARE preprocessing complete. Output saved to {output_dir}")


def preprocess_hrf(
    data_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe: bool = True,
    use_green_channel: bool = False,
) -> None:
    """
    Preprocess HRF (High Resolution Fundus) dataset for training.

    HRF dataset structure:
    - images/: XX_dr.JPG, XX_g.jpg, XX_h.jpg (45 images: 15 diabetic, 15 glaucoma, 15 healthy)
    - manual1/: XX_dr.tif, XX_g.tif, XX_h.tif (vessel annotations)
    - mask/: XX_dr_mask.tif, XX_g_mask.tif, XX_h_mask.tif (FOV masks)

    Args:
        data_dir: Path to HRF dataset root
        output_dir: Path to save preprocessed data
        target_size: Target image size (height, width)
        apply_clahe: Whether to apply CLAHE enhancement
        use_green_channel: Whether to use only green channel
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Create output directories
    out_images = output_path / "images"
    out_masks = output_path / "masks"
    out_fov = output_path / "fov"

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_fov.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_dir = data_path / "images"
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")))

    print(f"Processing HRF dataset ({len(image_files)} images)...")

    for img_path in tqdm(image_files):
        # Extract image ID (e.g., "01_dr" from "01_dr.JPG")
        img_id = img_path.stem

        # Load image
        image = np.array(Image.open(img_path))

        # Load vessel annotation
        manual_path = data_path / "manual1" / f"{img_id}.tif"
        if manual_path.exists():
            vessel_mask = np.array(Image.open(manual_path))
            vessel_mask = (vessel_mask > 0).astype(np.uint8)
        else:
            print(f"Warning: No label found for {img_id}")
            vessel_mask = None

        # Load FOV mask
        fov_path = data_path / "mask" / f"{img_id}_mask.tif"
        if fov_path.exists():
            fov_mask = np.array(Image.open(fov_path))
            # Convert to grayscale if RGB
            if len(fov_mask.shape) == 3:
                fov_mask = fov_mask[:, :, 0]
            fov_mask = (fov_mask > 0).astype(np.uint8)
        else:
            fov_mask = create_fov_mask(image)

        # Apply preprocessing
        if apply_clahe:
            image = clahe_enhancement(image)

        if use_green_channel:
            image = extract_green_channel(image)

        # Resize to target size
        image, fov_mask = resize_and_pad(image, target_size, fov_mask)
        if vessel_mask is not None:
            vessel_mask, _ = resize_and_pad(vessel_mask, target_size)

        # Save preprocessed data
        Image.fromarray(image).save(out_images / f"{img_id}.png")
        Image.fromarray((fov_mask * 255).astype(np.uint8)).save(out_fov / f"{img_id}.png")
        if vessel_mask is not None:
            Image.fromarray((vessel_mask * 255).astype(np.uint8)).save(out_masks / f"{img_id}.png")

    print(f"HRF preprocessing complete. Output saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess retinal vessel datasets")
    parser.add_argument("--dataset", choices=["drive", "stare", "stare-all", "hrf", "all"], default="all")
    parser.add_argument("--drive-dir", type=str, default="../data/DRIVE")
    parser.add_argument("--stare-dir", type=str, default="../data/STARE")
    parser.add_argument("--hrf-dir", type=str, default="../data/downloads/HRF_extracted")
    parser.add_argument("--output-dir", type=str, default="../data/processed")
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    parser.add_argument("--green-channel", action="store_true", help="Use only green channel")

    args = parser.parse_args()
    target_size = (args.size, args.size)

    if args.dataset in ["drive", "all"]:
        preprocess_drive(
            args.drive_dir,
            f"{args.output_dir}/DRIVE",
            target_size=target_size,
            apply_clahe=not args.no_clahe,
            use_green_channel=args.green_channel,
        )

    if args.dataset in ["stare", "all"]:
        # Only labeled images (20)
        preprocess_stare(
            args.stare_dir,
            f"{args.output_dir}/STARE",
            target_size=target_size,
            apply_clahe=not args.no_clahe,
            use_green_channel=args.green_channel,
            labeled_only=True,
        )

    if args.dataset in ["stare-all"]:
        # All 397 images (for preconditioning without labels)
        preprocess_stare(
            args.stare_dir,
            f"{args.output_dir}/STARE_ALL",
            target_size=target_size,
            apply_clahe=not args.no_clahe,
            use_green_channel=args.green_channel,
            labeled_only=False,
        )

    if args.dataset in ["hrf", "all"]:
        preprocess_hrf(
            args.hrf_dir,
            f"{args.output_dir}/HRF",
            target_size=target_size,
            apply_clahe=not args.no_clahe,
            use_green_channel=args.green_channel,
        )
