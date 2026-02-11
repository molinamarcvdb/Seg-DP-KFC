"""
Oxford-IIIT Pet Dataset wrapper for binary segmentation.

Easy to download, good size (~7400 images), suitable for DP experiments.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np
from PIL import Image


class OxfordPetSegmentation(Dataset):
    """
    Oxford-IIIT Pet dataset for binary segmentation.

    Converts trimap segmentation to binary (foreground/background).
    Resizes to specified size for efficient training.
    """

    def __init__(
        self,
        root: str = "./data/oxford_pet",
        split: str = "trainval",  # "trainval" or "test"
        image_size: int = 128,
        download: bool = True,
        augment: bool = False,
    ):
        """
        Args:
            root: Data directory
            split: "trainval" (3680 images) or "test" (3669 images)
            image_size: Resize images to this size
            download: Download if not present
            augment: Apply data augmentation
        """
        self.image_size = image_size
        self.augment = augment

        # Load dataset
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )

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
        ])

        # Augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, trimap = self.dataset[idx]

        # Apply augmentation (same transform to image and mask)
        if self.aug_transform is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            trimap = self.aug_transform(trimap)

        # Resize and normalize image
        image = self.img_transform(image)

        # Convert trimap to binary mask
        # Trimap: 1=foreground, 2=background, 3=boundary
        # Binary: 1=foreground (pet), 0=background
        trimap = self.mask_transform(trimap)
        trimap = torch.tensor(np.array(trimap), dtype=torch.long)
        mask = (trimap == 1).float().unsqueeze(0)  # [1, H, W]

        return image, mask


def get_pet_dataloaders(
    root: str = "./data/oxford_pet",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    download: bool = True,
):
    """Create train and test dataloaders for Oxford Pet segmentation."""
    from torch.utils.data import DataLoader

    train_dataset = OxfordPetSegmentation(
        root=root,
        split="trainval",
        image_size=image_size,
        download=download,
        augment=True,
    )

    test_dataset = OxfordPetSegmentation(
        root=root,
        split="test",
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


# Quick test
if __name__ == "__main__":
    print("Testing Oxford Pet Segmentation...")
    dataset = OxfordPetSegmentation(download=True, image_size=128)
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
