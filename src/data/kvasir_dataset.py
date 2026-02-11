"""
Kvasir-SEG Dataset for polyp segmentation.

Easy to download (~46MB), 1000 images with masks.
Good for DP experiments - medical domain with fine-grained structures.
"""

import os
import ssl
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import urllib.request


KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_kvasir(root: str = "./data/kvasir_seg", force: bool = False) -> Path:
    """
    Download and extract Kvasir-SEG dataset.

    Args:
        root: Directory to save dataset
        force: Re-download even if exists

    Returns:
        Path to extracted dataset
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "kvasir-seg.zip"
    data_path = root / "Kvasir-SEG"

    # Check if already downloaded
    if data_path.exists() and not force:
        print(f"Kvasir-SEG already exists at {data_path}")
        return data_path

    # Download with SSL handling (disable verification for HPC environments)
    print(f"Downloading Kvasir-SEG from {KVASIR_URL}...")
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context)
    )
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(KVASIR_URL, zip_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Kvasir-SEG. Please download manually from:\n"
            f"  {KVASIR_URL}\n"
            f"and extract to {root}/Kvasir-SEG/\n"
            f"Error: {e}"
        )
    print("Download complete.")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    print(f"Extracted to {data_path}")

    # Clean up zip
    zip_path.unlink()

    return data_path


class KvasirSegDataset(Dataset):
    """
    Kvasir-SEG dataset for polyp segmentation.

    1000 images of gastrointestinal polyps with ground truth masks.
    """

    def __init__(
        self,
        root: str = "./data/kvasir_seg",
        split: str = "train",  # "train" (800) or "val" (200)
        image_size: int = 256,
        download: bool = True,
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            root: Data directory
            split: "train" or "val"
            image_size: Resize images to this size
            download: Download if not present
            augment: Apply data augmentation
            seed: Random seed for train/val split
        """
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        # Download if needed
        if download:
            data_path = download_kvasir(root)
        else:
            data_path = self.root / "Kvasir-SEG"

        self.images_dir = data_path / "images"
        self.masks_dir = data_path / "masks"

        # Get all image files
        all_images = sorted(list(self.images_dir.glob("*.jpg")))

        # Split train/val (80/20)
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        split_idx = int(0.8 * len(all_images))

        if split == "train":
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        self.image_files = [all_images[i] for i in selected_indices]

        # Transforms
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

        # Augmentation
        if augment:
            self.aug_hflip = transforms.RandomHorizontalFlip(p=0.5)
            self.aug_vflip = transforms.RandomVerticalFlip(p=0.5)
            self.aug_rotate = transforms.RandomRotation(15)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.aug_hflip(image)
            torch.manual_seed(seed)
            mask = self.aug_hflip(mask)

            torch.manual_seed(seed + 1)
            image = self.aug_vflip(image)
            torch.manual_seed(seed + 1)
            mask = self.aug_vflip(mask)

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        return image, mask


def get_kvasir_dataloaders(
    root: str = "./data/kvasir_seg",
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    download: bool = True,
):
    """Create train and val dataloaders for Kvasir-SEG."""
    from torch.utils.data import DataLoader

    train_dataset = KvasirSegDataset(
        root=root,
        split="train",
        image_size=image_size,
        download=download,
        augment=True,
    )

    val_dataset = KvasirSegDataset(
        root=root,
        split="val",
        image_size=image_size,
        download=download,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Kvasir-SEG Dataset...")
    dataset = KvasirSegDataset(download=True, image_size=256)
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
