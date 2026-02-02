"""
Classification Models for MedMNIST.

Simple CNN architectures suitable for DP-SGD training:
- Uses GroupNorm instead of BatchNorm (DP-compatible)
- Configurable depth and width
- Supports 2D (28x28) and 3D inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock2D(nn.Module):
    """Conv -> GroupNorm -> ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_groups: int = 8,
    ):
        super().__init__()
        # Ensure num_groups divides out_channels
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=False)  # Opacus requires inplace=False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class SimpleCNN(nn.Module):
    """
    Simple CNN for MedMNIST classification.

    Architecture:
    - 3-4 conv blocks with pooling
    - Global average pooling
    - Fully connected output

    DP-compatible (uses GroupNorm).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        features: List[int] = [32, 64, 128],
        input_size: int = 28,
    ):
        """
        Args:
            in_channels: Number of input channels (1 or 3)
            num_classes: Number of output classes
            features: List of channel sizes for conv blocks
            input_size: Input image size (28 for MedMNIST)
        """
        super().__init__()

        self.features = nn.ModuleList()

        prev_channels = in_channels
        for feat in features:
            self.features.append(ConvBlock2D(prev_channels, feat))
            self.features.append(nn.MaxPool2d(2, 2))
            prev_channels = feat

        # Calculate feature map size after pooling
        # For 28x28 with 3 pooling layers: 28 -> 14 -> 7 -> 3
        n_pools = len(features)
        final_size = input_size // (2 ** n_pools)
        final_size = max(final_size, 1)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MedMNISTCNN(nn.Module):
    """
    CNN specifically designed for MedMNIST 28x28 images.

    Slightly deeper architecture with residual-style connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        base_channels: int = 32,
    ):
        super().__init__()

        # Block 1: 28x28 -> 14x14
        self.block1 = nn.Sequential(
            ConvBlock2D(in_channels, base_channels),
            ConvBlock2D(base_channels, base_channels),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 14x14 -> 7x7
        self.block2 = nn.Sequential(
            ConvBlock2D(base_channels, base_channels * 2),
            ConvBlock2D(base_channels * 2, base_channels * 2),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 7x7 -> 3x3
        self.block3 = nn.Sequential(
            ConvBlock2D(base_channels * 2, base_channels * 4),
            ConvBlock2D(base_channels * 4, base_channels * 4),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LargeCNN(nn.Module):
    """
    Larger CNN for MedMNIST classification.

    Deeper architecture with more channels (~2.3M parameters).
    DP-compatible (uses GroupNorm, no inplace ops).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        base_channels: int = 64,
    ):
        super().__init__()

        # Block 1: 28x28 -> 14x14 (3 conv layers)
        self.block1 = nn.Sequential(
            ConvBlock2D(in_channels, base_channels),
            ConvBlock2D(base_channels, base_channels),
            ConvBlock2D(base_channels, base_channels),
            nn.MaxPool2d(2, 2),
        )

        # Block 2: 14x14 -> 7x7 (3 conv layers)
        self.block2 = nn.Sequential(
            ConvBlock2D(base_channels, base_channels * 2),
            ConvBlock2D(base_channels * 2, base_channels * 2),
            ConvBlock2D(base_channels * 2, base_channels * 2),
            nn.MaxPool2d(2, 2),
        )

        # Block 3: 7x7 -> 3x3 (3 conv layers)
        self.block3 = nn.Sequential(
            ConvBlock2D(base_channels * 2, base_channels * 4),
            ConvBlock2D(base_channels * 4, base_channels * 4),
            ConvBlock2D(base_channels * 4, base_channels * 4),
            nn.MaxPool2d(2, 2),
        )

        # Block 4: 3x3 -> 1x1 (3 conv layers)
        self.block4 = nn.Sequential(
            ConvBlock2D(base_channels * 4, base_channels * 8),
            ConvBlock2D(base_channels * 8, base_channels * 8),
            ConvBlock2D(base_channels * 8, base_channels * 8),
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SimpleCNN3D(nn.Module):
    """
    Simple 3D CNN for MedMNIST3D datasets.

    Architecture similar to 2D version but with 3D convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 11,
        features: List[int] = [32, 64, 128],
    ):
        super().__init__()

        self.features = nn.ModuleList()

        prev_channels = in_channels
        for feat in features:
            # Ensure num_groups divides feat
            num_groups = min(8, feat)
            while feat % num_groups != 0:
                num_groups -= 1

            self.features.append(
                nn.Sequential(
                    nn.Conv3d(prev_channels, feat, 3, padding=1, bias=False),
                    nn.GroupNorm(num_groups, feat),
                    nn.ReLU(inplace=False),  # Opacus requires inplace=False
                )
            )
            self.features.append(nn.MaxPool3d(2, 2))
            prev_channels = feat

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(features[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_classifier(
    name: str = "simple_cnn",
    in_channels: int = 3,
    num_classes: int = 9,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create classification model.

    Args:
        name: Model name ("simple_cnn", "medmnist_cnn", "simple_cnn_3d")
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models

    Returns:
        Model instance
    """
    if name == "simple_cnn":
        return SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            features=kwargs.get("features", [32, 64, 128]),
            input_size=kwargs.get("input_size", 28),
        )
    elif name == "medmnist_cnn":
        return MedMNISTCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=kwargs.get("base_channels", 32),
        )
    elif name == "large_cnn":
        return LargeCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=kwargs.get("base_channels", 64),
        )
    elif name == "simple_cnn_3d":
        return SimpleCNN3D(
            in_channels=in_channels,
            num_classes=num_classes,
            features=kwargs.get("features", [32, 64, 128]),
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# Quick test
if __name__ == "__main__":
    # Test 2D model
    print("Testing SimpleCNN (2D)...")
    model = SimpleCNN(in_channels=3, num_classes=9)
    x = torch.randn(4, 3, 28, 28)
    y = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test MedMNIST CNN
    print("\nTesting MedMNISTCNN...")
    model = MedMNISTCNN(in_channels=3, num_classes=9)
    y = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 3D model
    print("\nTesting SimpleCNN3D...")
    model = SimpleCNN3D(in_channels=1, num_classes=11)
    x = torch.randn(2, 1, 28, 28, 28)
    y = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
