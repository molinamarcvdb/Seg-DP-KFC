"""
U-Net for Retinal Vessel Segmentation.

A simple, clean U-Net implementation suitable for DP-SGD training.
Avoids batch normalization (incompatible with per-sample gradients)
and uses group normalization instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm (DP-compatible)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides out_channels
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + Concat + ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (padding if needed)
        if x.shape != skip.shape:
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for segmentation.

    Features:
    - GroupNorm instead of BatchNorm (DP-compatible)
    - Configurable depth and channel sizes
    - Skip connections for multi-scale features
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            features: List of channel sizes for each encoder level
        """
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder path
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path (reverse order)
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2

        for i, feat in enumerate(reversed_features):
            self.decoders.append(
                DecoderBlock(prev_channels, feat, feat)
            )
            prev_channels = feat

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (use skip connections in reverse)
        skip_connections = skip_connections[::-1]
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip)

        return self.final_conv(x)


class UNetSmall(nn.Module):
    """
    Smaller U-Net for faster training and lower memory.

    Good for initial experiments and debugging.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        # Smaller feature sizes
        features = [16, 32, 64, 128]

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat))
            prev_channels = feat

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2

        for feat in reversed_features:
            self.decoders.append(DecoderBlock(prev_channels, feat, feat))
            prev_channels = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip)

        return self.final_conv(x)


def create_model(
    name: str = "unet",
    in_channels: int = 3,
    out_channels: int = 1,
    features: List[int] = [32, 64, 128, 256],
) -> nn.Module:
    """
    Factory function to create segmentation model.

    Args:
        name: "unet" or "unet_small"
        in_channels: Input channels
        out_channels: Output channels
        features: Channel sizes (only for "unet")

    Returns:
        Model instance
    """
    if name == "unet":
        return UNet(in_channels, out_channels, features)
    elif name == "unet_small":
        return UNetSmall(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model: {name}")


# Quick test
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
