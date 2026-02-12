"""
Retinal Vessel Segmentation Dataset (FIVES-like).

Structure:
    root/
        train/Original/*.png       (RGB fundus images, 2048x2048)
        train/Ground truth/*.png   (Binary vessel masks, RGB with 0/255)
        test/Original/*.png
        test/Ground truth/*.png

Binary target: vessel pixels (any channel > 0).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


RETINAL_DEFAULT_ROOT = "./data/archive-2"


def _discover_retinal_pairs(root: str, folder: str = "train"):
    """Discover image/mask pairs in the given folder."""
    root = Path(root) / folder
    images_dir = root / "Original"
    masks_dir = root / "Ground truth"

    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        if img_path.name == "Thumbs.db":
            continue
        mask_path = masks_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


class RetinalVesselDataset(Dataset):
    """
    Retinal vessel segmentation dataset.

    Loads RGB fundus images and binary vessel masks.
    Uses train folder with 80/20 split for train/val.
    """

    def __init__(
        self,
        root: str = RETINAL_DEFAULT_ROOT,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.augment = augment

        all_pairs = _discover_retinal_pairs(root, folder="train")
        if len(all_pairs) == 0:
            raise FileNotFoundError(
                f"No retinal vessel images found in {root}/train. "
                f"Expected train/Original/ and train/Ground truth/ subdirectories."
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

        # Binary vessel mask from any channel
        mask_arr = np.array(mask_rgb)[:, :, 0]
        binary = (mask_arr > 0).astype(np.uint8) * 255
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


if __name__ == "__main__":
    print("Testing Retinal Vessel Dataset...")
    dataset = RetinalVesselDataset(image_size=256)
    print(f"Train size: {len(dataset)}")

    val = RetinalVesselDataset(split="val", image_size=256)
    print(f"Val size: {len(val)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
    print(f"Vessel ratio: {mask.mean():.4f}")
