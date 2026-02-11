"""
3D U-Net for Brain Tumor Segmentation (BraTS).

Mirrors unet.py exactly, replacing 2D ops with 3D equivalents.
Uses GroupNorm (DP-compatible, dimension-agnostic).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock3D(nn.Module):
    """Double convolution block with GroupNorm (DP-compatible)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class EncoderBlock3D(nn.Module):
    """Encoder block: ConvBlock3D + MaxPool3d."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock3D(nn.Module):
    """Decoder block: Upsample + Concat + ConvBlock3D."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock3D(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch (padding if needed)
        if x.shape != skip.shape:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Features:
    - GroupNorm instead of BatchNorm (DP-compatible)
    - Configurable depth and channel sizes
    - Skip connections for multi-scale features
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: List[int] = [16, 32, 64, 128],
    ):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder path
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock3D(prev_channels, feat))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # Decoder path (reverse order)
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2

        for i, feat in enumerate(reversed_features):
            self.decoders.append(
                DecoderBlock3D(prev_channels, feat, feat)
            )
            prev_channels = feat

        # Final output
        self.final_conv = nn.Conv3d(features[0], out_channels, 1)

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


def create_model_3d(
    name: str = "unet3d",
    in_channels: int = 4,
    out_channels: int = 1,
    features: List[int] = [16, 32, 64, 128],
) -> nn.Module:
    """
    Factory function to create 3D segmentation model.

    Args:
        name: "unet3d"
        in_channels: Input channels (4 for BraTS MRI modalities)
        out_channels: Output channels (1 for binary segmentation)
        features: Channel sizes per encoder level

    Returns:
        Model instance
    """
    if name == "unet3d":
        return UNet3D(in_channels, out_channels, features)
    else:
        raise ValueError(f"Unknown 3D model: {name}")


# Quick test
if __name__ == "__main__":
    model = UNet3D(in_channels=4, out_channels=1)
    x = torch.randn(2, 4, 128, 128, 128)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
