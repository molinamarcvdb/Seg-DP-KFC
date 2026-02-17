"""STU-Net-S architecture adapted for Opacus (DP-SGD) compatibility.

Based on: https://github.com/uni-medical/STU-Net
Paper: "STU-Net: Scalable and Transferable Medical Image Segmentation Models
        Empowered by Large-Scale Supervised Pre-training" (arXiv:2304.06716)

Key modifications for Opacus:
1. InstanceNorm3d → GroupNorm(C, C) — mathematically equivalent, Opacus-compatible
2. LeakyReLU(inplace=True) → LeakyReLU(inplace=False)
3. Removed SegmentationNetwork base class dependency (nnUNet-specific)
4. Removed deep supervision (not needed for our training loop)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    """Residual block with two 3D convolutions + GroupNorm."""

    def __init__(self, input_channels, output_channels,
                 kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels,
                               kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.GroupNorm(output_channels, output_channels)
        self.act1 = nn.LeakyReLU(inplace=False)

        self.conv2 = nn.Conv3d(output_channels, output_channels,
                               kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(output_channels, output_channels)
        self.act2 = nn.LeakyReLU(inplace=False)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UpsampleLayer(nn.Module):
    """Nearest-neighbor upsample + 1x1 conv for channel reduction."""

    def __init__(self, input_channels, output_channels, scale_factor, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=list(self.scale_factor), mode=self.mode)
        x = self.conv(x)
        return x


class STUNet(nn.Module):
    """STU-Net: Scalable and Transferable U-Net for 3D medical segmentation.

    This is a standalone version (no nnUNet dependency) with Opacus compatibility.

    Args:
        input_channels: Number of input channels (1 for CT).
        num_classes: Number of output classes.
        depth: Number of BasicResBlocks per stage.
        dims: Channel dimensions per stage.
        pool_op_kernel_sizes: Downsampling factors between stages.
        conv_kernel_sizes: Convolution kernel sizes per stage.
    """

    # STU-Net-S default configuration
    SMALL_CONFIG = {
        "depth": [1, 1, 1, 1, 1, 1],
        "dims": [16, 32, 64, 128, 256, 256],
        "pool_op_kernel_sizes": [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    }

    def __init__(self, input_channels=1, num_classes=1,
                 depth=None, dims=None,
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None):
        super().__init__()

        # Default to STU-Net-S config
        cfg = self.SMALL_CONFIG
        depth = depth or cfg["depth"]
        dims = dims or cfg["dims"]
        pool_op_kernel_sizes = pool_op_kernel_sizes or cfg["pool_op_kernel_sizes"]
        conv_kernel_sizes = conv_kernel_sizes or cfg["conv_kernel_sizes"]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dims = dims

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        conv_pad_sizes = [[k // 2 for k in krnl] for krnl in conv_kernel_sizes]

        # Encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0],
                          conv_kernel_sizes[0], conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], conv_kernel_sizes[0], conv_pad_sizes[0])
              for _ in range(depth[0] - 1)]
        )
        self.conv_blocks_context.append(stage)

        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d], conv_kernel_sizes[d], conv_pad_sizes[d],
                              stride=pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], conv_kernel_sizes[d], conv_pad_sizes[d])
                  for _ in range(depth[d] - 1)]
            )
            self.conv_blocks_context.append(stage)

        # Upsample layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            self.upsample_layers.append(
                UpsampleLayer(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            )

        # Decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2, dims[-2 - u],
                              conv_kernel_sizes[-2 - u], conv_pad_sizes[-2 - u], use_1x1conv=True),
                *[BasicResBlock(dims[-2 - u], dims[-2 - u],
                                conv_kernel_sizes[-2 - u], conv_pad_sizes[-2 - u])
                  for _ in range(depth[-2 - u] - 1)]
            )
            self.conv_blocks_localization.append(stage)

        # Segmentation outputs (deep supervision heads)
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass returning only the final segmentation output."""
        skips = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

        return self.seg_outputs[-1](x)

    def forward_encoder(self, x):
        """Forward pass through encoder only, returning multi-scale features.

        Returns:
            features: list of tensors at each encoder stage
                      [stage0, stage1, ..., bottleneck]
        """
        features = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            features.append(x)

        x = self.conv_blocks_context[-1](x)
        features.append(x)

        return features


def load_stunet_pretrained(model, weights_path, strict=False):
    """Load pretrained STU-Net weights.

    Handles:
    - 'module.' prefix from DataParallel
    - InstanceNorm3d → GroupNorm weight name mapping
    - Skipping seg_outputs (task-specific heads)
    - Input channel adaptation (repeat if needed)

    Args:
        model: STUNet instance
        weights_path: Path to pretrained .model file
        strict: Whether to require exact key matching
    """
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip DataParallel prefix
        if k.startswith('module.'):
            k = k[7:]

        # Skip seg_outputs (task-specific)
        if k.startswith('seg_outputs'):
            continue

        # Map InstanceNorm3d param names to GroupNorm
        # InstanceNorm3d: .weight, .bias → GroupNorm: .weight, .bias (same names)
        # No renaming needed — the param names are identical

        new_state_dict[k] = v

    # Handle input channel mismatch
    first_conv_key = 'conv_blocks_context.0.0.conv1.weight'
    if first_conv_key in new_state_dict:
        pretrained_in_ch = new_state_dict[first_conv_key].shape[1]
        model_in_ch = model.input_channels
        if pretrained_in_ch != model_in_ch:
            print(f"  Adapting input channels: {pretrained_in_ch} → {model_in_ch}")
            w = new_state_dict[first_conv_key]
            # Average across input channels then repeat
            w_mean = w.mean(dim=1, keepdim=True)
            new_state_dict[first_conv_key] = w_mean.repeat(1, model_in_ch, 1, 1, 1)

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded pretrained weights from {weights_path}")
    print(f"  Missing keys: {len(missing)} (expected: seg_outputs)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model
