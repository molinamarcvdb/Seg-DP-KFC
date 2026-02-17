"""Frozen STU-Net-S encoder with lightweight segmentation head for DP-SGD.

Mirrors the RADIOLoRA pattern: freeze a large pretrained backbone and only
train a small segmentation head (~1.5-2M params), keeping DP noise manageable.

Architecture:
    Input (B, 1, D, H, W)
      → STU-Net-S encoder (FROZEN, ~7M) → multi-scale features
      → FPN-style seg head (TRAINABLE, ~1.5M) → logits (B, C, D, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stunet import STUNet, load_stunet_pretrained


class SegHead3D(nn.Module):
    """Lightweight FPN-style 3D segmentation head.

    Takes multi-scale features from encoder stages 2-5 (channels 64, 128, 256, 256),
    projects to a common channel dim, upsamples and fuses, then predicts.

    Architecture:
        Each feature level → Conv3d(C_in, mid_ch, 1) + GroupNorm + ReLU
        → Upsample all to stage-2 resolution (1/4 of input)
        → Sum fusion
        → Conv3d(mid_ch, mid_ch, 3) + GroupNorm + ReLU
        → Upsample to full resolution
        → Conv3d(mid_ch, num_classes, 1)

    Uses GroupNorm and ReLU(inplace=False) for Opacus compatibility.
    """

    def __init__(self, encoder_dims, num_classes=1, mid_ch=64):
        super().__init__()
        self.num_classes = num_classes
        # encoder_dims: e.g. [16, 32, 64, 128, 256, 256] for STU-Net-S
        # We use stages 2-5 (indices 2,3,4,5): dims 64, 128, 256, 256

        self.lateral_convs = nn.ModuleList()
        for dim in encoder_dims[2:]:  # skip first 2 low-level stages
            self.lateral_convs.append(nn.Sequential(
                nn.Conv3d(dim, mid_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(16, mid_ch), mid_ch),
                nn.ReLU(inplace=False),
            ))

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.ReLU(inplace=False),
        )

        self.out_conv = nn.Conv3d(mid_ch, num_classes, kernel_size=1)

    def forward(self, features, output_size):
        """
        Args:
            features: list of encoder features [stage0, ..., stage5]
            output_size: target spatial size (D, H, W)
        """
        # Use stages 2-5
        feats = features[2:]

        # Lateral projections
        projected = []
        for feat, lat_conv in zip(feats, self.lateral_convs):
            projected.append(lat_conv(feat))

        # Upsample all to the resolution of stage 2 (highest res among selected)
        target_size = projected[0].shape[2:]
        fused = projected[0]
        for p in projected[1:]:
            fused = fused + F.interpolate(p, size=target_size, mode='nearest')

        x = self.fuse_conv(fused)

        # Upsample to full input resolution
        x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=False)
        x = self.out_conv(x)

        return x


class STUNetSeg(nn.Module):
    """Frozen STU-Net-S encoder with trainable segmentation head.

    Args:
        weights_path: Path to pretrained STU-Net-S weights (.model file).
        num_classes: Number of segmentation classes (1 for binary).
        mid_ch: Hidden channels in the seg head.
        device: Device string.
    """

    def __init__(self, weights_path=None, num_classes=1, mid_ch=64, device='cpu'):
        super().__init__()
        self._device = device

        # Build STU-Net-S encoder
        self.encoder = STUNet(input_channels=1, num_classes=num_classes)

        # Load pretrained weights
        if weights_path is not None:
            load_stunet_pretrained(self.encoder, weights_path)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Segmentation head (trainable)
        self.seg_head = SegHead3D(
            encoder_dims=self.encoder.dims,
            num_classes=num_classes,
            mid_ch=mid_ch,
        )

        self.to(device)

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) CT volume patches.
        Returns:
            (B, num_classes, D, H, W) segmentation logits.
        """
        output_size = x.shape[2:]

        # Frozen encoder forward
        with torch.no_grad():
            features = self.encoder.forward_encoder(x)

        # Detach features so no gradients flow to encoder
        features = [f.detach() for f in features]

        # Trainable seg head
        logits = self.seg_head(features, output_size)
        return logits

    def get_trainable_params(self):
        """Iterator over trainable parameters (seg_head only)."""
        for p in self.seg_head.parameters():
            if p.requires_grad:
                yield p

    def count_params(self):
        """Count trainable and total parameters."""
        seg_head_params = sum(p.numel() for p in self.seg_head.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        return {
            "seg_head": seg_head_params,
            "total_trainable": seg_head_params,
            "encoder_frozen": encoder_params,
        }
