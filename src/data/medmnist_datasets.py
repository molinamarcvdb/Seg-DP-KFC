"""
MedMNIST Dataset Wrappers for DP-AdaDPS Experiments.

Provides unified interface for MedMNIST 2D and 3D datasets with support for:
- Classification tasks (multi-class and multi-label)
- DP-compatible data loading
- Train/val/test splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Literal
import medmnist
from medmnist import INFO


# Available 2D datasets
MEDMNIST_2D = [
    "pathmnist",      # 107,180 - 9-class pathology
    "dermamnist",     # 10,015 - 7-class dermatology
    "bloodmnist",     # 17,092 - 8-class blood cells
    "chestmnist",     # 112,120 - 14-label chest X-ray (multi-label)
    "organamnist",    # 58,850 - 11-class CT organs (axial)
    "organcmnist",    # 23,583 - 11-class CT organs (coronal)
    "organsmnist",    # 25,221 - 11-class CT organs (sagittal)
    "pneumoniamnist", # 5,856 - 2-class pneumonia
    "retinamnist",    # 1,600 - 5-class diabetic retinopathy
    "breastmnist",    # 780 - 2-class breast ultrasound
    "tissuemnist",    # 236,386 - 8-class kidney cortex
    "octmnist",       # 109,309 - 4-class retinal OCT
]

# Available 3D datasets
MEDMNIST_3D = [
    "organmnist3d",   # 1,743 - 11-class CT organs
    "nodulemnist3d",  # 1,633 - 2-class lung nodule
    "adrenalmnist3d", # 1,584 - 2-class adrenal
    "fracturemnist3d",# 1,370 - 3-class fracture
    "vesselmnist3d",  # 1,909 - 2-class vessel
    "synapsemnist3d", # 1,759 - 2-class synapse
]


class MedMNISTDataset(Dataset):
    """
    Wrapper for MedMNIST datasets with unified interface.

    Features:
    - Automatic download
    - Normalized to [0, 1]
    - Optional transforms
    - Returns (image, label) tuples
    """

    def __init__(
        self,
        name: str,
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[Callable] = None,
        download: bool = True,
        root: str = "./data/medmnist",
        as_rgb: bool = True,
    ):
        """
        Args:
            name: Dataset name (e.g., "pathmnist", "dermamnist")
            split: Data split ("train", "val", "test")
            transform: Optional transform to apply
            download: Whether to download if not present
            root: Root directory for data
            as_rgb: Convert grayscale to RGB (3 channels)
        """
        self.name = name.lower()
        self.split = split
        self.transform = transform
        self.as_rgb = as_rgb

        # Get dataset info
        if self.name not in INFO:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(INFO.keys())}")

        self.info = INFO[self.name]
        self.n_classes = len(self.info["label"])
        self.is_multilabel = self.info["task"] == "multi-label, binary-class"
        self.is_3d = "3d" in self.name

        # Get the dataset class
        dataset_class = getattr(medmnist, self.info["python_class"])

        # Load dataset
        self.dataset = dataset_class(
            split=split,
            transform=None,  # We apply transforms ourselves
            download=download,
            root=root,
            as_rgb=as_rgb,
        )

        # Store images and labels as numpy arrays
        self.images = self.dataset.imgs  # Shape: [N, H, W] or [N, H, W, C] or [N, D, H, W]
        self.labels = self.dataset.labels  # Shape: [N, 1] or [N, num_labels]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Handle channel dimension
        if self.is_3d:
            # 3D: [D, H, W] -> [1, D, H, W]
            if image.ndim == 3:
                image = image[np.newaxis, ...]
        else:
            # 2D: ensure [H, W, C] format for transforms
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
                if self.as_rgb:
                    image = np.repeat(image, 3, axis=2)

        # Apply transforms (expects HWC for 2D)
        if self.transform is not None:
            if self.is_3d:
                # For 3D, transform expects [D, H, W] or [C, D, H, W]
                image = self.transform(image)
            else:
                # For 2D albumentations-style transforms
                if hasattr(self.transform, '__call__'):
                    if isinstance(self.transform, Callable):
                        # Check if it's albumentations
                        try:
                            transformed = self.transform(image=image)
                            image = transformed["image"]
                        except (TypeError, KeyError):
                            # Fallback to direct transform
                            image = self.transform(image)

        # Convert to tensor
        if isinstance(image, np.ndarray):
            if self.is_3d:
                # 3D: already [C, D, H, W]
                image = torch.from_numpy(image).float()
            else:
                # 2D: [H, W, C] -> [C, H, W]
                image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Handle labels
        if self.is_multilabel:
            label = torch.from_numpy(label).float().squeeze()
        else:
            label = torch.tensor(label.item(), dtype=torch.long)

        return image, label

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def task_type(self) -> str:
        return "multilabel" if self.is_multilabel else "multiclass"

    def __repr__(self) -> str:
        return (
            f"MedMNISTDataset(name={self.name}, split={self.split}, "
            f"n_samples={len(self)}, n_classes={self.n_classes}, "
            f"task={self.task_type})"
        )


def get_medmnist_dataset(
    name: str,
    split: str = "train",
    transform: Optional[Callable] = None,
    download: bool = True,
    root: str = "./data/medmnist",
) -> MedMNISTDataset:
    """
    Factory function to create MedMNIST dataset.

    Args:
        name: Dataset name (e.g., "pathmnist", "dermamnist")
        split: Data split ("train", "val", "test")
        transform: Optional transform
        download: Whether to download
        root: Root directory

    Returns:
        MedMNISTDataset instance
    """
    return MedMNISTDataset(
        name=name,
        split=split,
        transform=transform,
        download=download,
        root=root,
    )


def get_medmnist_info(name: str) -> dict:
    """Get dataset information."""
    if name.lower() not in INFO:
        raise ValueError(f"Unknown dataset: {name}")
    return INFO[name.lower()]


def list_available_datasets() -> dict:
    """List all available MedMNIST datasets with sizes."""
    result = {"2d": {}, "3d": {}}

    for name in MEDMNIST_2D:
        info = INFO[name]
        result["2d"][name] = {
            "n_train": info["n_samples"]["train"],
            "n_val": info["n_samples"]["val"],
            "n_test": info["n_samples"]["test"],
            "n_classes": len(info["label"]),
            "task": info["task"],
        }

    for name in MEDMNIST_3D:
        info = INFO[name]
        result["3d"][name] = {
            "n_train": info["n_samples"]["train"],
            "n_val": info["n_samples"]["val"],
            "n_test": info["n_samples"]["test"],
            "n_classes": len(info["label"]),
            "task": info["task"],
        }

    return result


# Quick test
if __name__ == "__main__":
    # Test PathMNIST
    print("Testing PathMNIST...")
    dataset = MedMNISTDataset("pathmnist", split="train", download=True)
    print(dataset)

    image, label = dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label}")

    # List all datasets
    print("\nAvailable datasets:")
    all_datasets = list_available_datasets()
    for dim, datasets in all_datasets.items():
        print(f"\n{dim.upper()}:")
        for name, info in datasets.items():
            print(f"  {name}: {info['n_train']} train, {info['n_classes']} classes")
