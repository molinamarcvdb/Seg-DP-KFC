"""Kvasir-SEG dataset with [0,1] normalization for RADIO backbone.

RADIO expects images in [0, 1] range (no ImageNet normalization).
This module provides a dataset compatible with RADIO-based models.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .kvasir_dataset import download_kvasir


class KvasirRADIODataset(Dataset):
    """Kvasir-SEG dataset normalized for RADIO ([0,1] range).

    Args:
        root: Data directory.
        split: "train" or "val".
        image_size: Resize images to this size (352 recommended for RADIO).
        download: Download if not present.
        augment: Apply data augmentation.
        seed: Random seed for train/val split.
    """

    def __init__(
        self,
        root: str = "./data/kvasir_seg",
        split: str = "train",
        image_size: int = 352,
        download: bool = True,
        augment: bool = False,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        if download:
            data_path = download_kvasir(root)
        else:
            data_path = self.root / "Kvasir-SEG"

        self.images_dir = data_path / "images"
        self.masks_dir = data_path / "masks"

        all_images = sorted(list(self.images_dir.glob("*.jpg")))

        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        split_idx = int(0.8 * len(all_images))

        if split == "train":
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        self.image_files = [all_images[i] for i in selected_indices]

        # RADIO expects [0, 1] — just ToTensor (no ImageNet normalization)
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        if augment:
            self.aug_hflip = transforms.RandomHorizontalFlip(p=0.5)
            self.aug_vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

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

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask


def get_kvasir_radio_dataloaders(
    root: str = "./data/kvasir_seg",
    image_size: int = 352,
    batch_size: int = 8,
    num_workers: int = 0,
    download: bool = True,
    seed: int = 42,
):
    """Create train and val dataloaders for Kvasir-SEG (RADIO format)."""
    train_ds = KvasirRADIODataset(
        root=root, split="train", image_size=image_size,
        download=download, augment=True, seed=seed,
    )
    val_ds = KvasirRADIODataset(
        root=root, split="val", image_size=image_size,
        download=download, augment=False, seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader
