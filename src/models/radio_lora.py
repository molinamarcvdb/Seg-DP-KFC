"""RADIO backbone with LoRA adapters for DP-SGD finetuning.

Adapted from MedRadSeg (fedrad/models/radio_lora.py, seg_head.py).

Key changes for DP (Opacus) compatibility:
1. LoRALinear uses nn.Linear submodules (not raw Parameters) so Opacus
   can compute per-sample gradients via its standard hooks.
2. SegmentationHead uses GroupNorm instead of BatchNorm.
3. FFA-LoRA variant: freeze lora_down (A), train only lora_up (B).
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA layer (Opacus-compatible)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer augmented with LoRA: y = Wx + scaling * lora_up(lora_down(x)).

    Uses nn.Linear submodules instead of raw Parameters so that Opacus
    GradSampleModule can register per-sample gradient hooks on them.

    Args:
        original_linear: Frozen pretrained nn.Linear.
        rank: LoRA rank r.
        alpha: LoRA scaling factor.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # LoRA adapters as nn.Linear (Opacus-friendly)
        self.lora_down = nn.Linear(d_in, rank, bias=False)   # A: (r, d_in)
        self.lora_up = nn.Linear(rank, d_out, bias=False)    # B: (d_out, r)

        # Initialize: down = Kaiming, up = zero (starts as identity)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return base_out + lora_out


# ---------------------------------------------------------------------------
# Segmentation head (GroupNorm for Opacus)
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Lightweight decoder: reshape patch tokens -> upsample -> segment.

    Architecture:
        patch_tokens (B, N, D) -> reshape (B, D, h, w)
        -> Conv2d(D, 256, 3) + GroupNorm + ReLU
        -> Upsample 4x
        -> Conv2d(256, 64, 3) + GroupNorm + ReLU
        -> Upsample 4x
        -> Conv2d(64, num_classes, 1)

    Total 16x upsampling to match ViT patch_size=16.
    Uses GroupNorm (not BatchNorm) for Opacus compatibility.
    """

    def __init__(self, embed_dim: int, num_classes: int = 1, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),  # 32 groups for 256 channels
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),   # 16 groups for 64 channels
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        B, N, D = patch_tokens.shape
        h = w = int(N**0.5)

        x = patch_tokens.transpose(1, 2).contiguous().reshape(B, D, h, w)
        x = self.decoder(x)

        if x.shape[-2:] != img_size:
            x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)

        return x


# ---------------------------------------------------------------------------
# RADIO backbone loading + LoRA injection
# ---------------------------------------------------------------------------

_shared_radio = None
_shared_radio_version = None


def _load_shared_radio(version: str, device: str):
    """Load RADIO backbone once and cache globally."""
    global _shared_radio, _shared_radio_version
    if _shared_radio is not None and _shared_radio_version == version:
        return _shared_radio

    print(f"Loading RADIO backbone ({version})...")
    radio = torch.hub.load(
        "NVlabs/RADIO", "radio_model",
        version=version, progress=True, skip_validation=True,
    )
    radio.to(device)
    radio.eval()
    for param in radio.parameters():
        param.requires_grad = False

    _shared_radio = radio
    _shared_radio_version = version
    return radio


def _get_attn_qkv(block) -> nn.Linear:
    return block.attn.qkv


def _set_attn_qkv(block, module):
    block.attn.qkv = module


class RADIOLoRA(nn.Module):
    """Frozen RADIO ViT with LoRA on the last N attention QKV projections.

    The backbone is loaded once and shared. Each instance only owns
    its LoRA parameters + segmentation head.

    Args:
        radio_version: RADIO model version (e.g. "c-radio_v3-b").
        lora_rank: LoRA rank r.
        lora_alpha: LoRA scaling alpha.
        num_lora_layers: How many last attention blocks get LoRA.
        num_classes: Segmentation classes (1 = binary).
        device: Device string.
        ffa_mode: If True, freeze lora_down (FFA-LoRA).
    """

    def __init__(
        self,
        radio_version: str = "c-radio_v3-b",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        num_lora_layers: int = 4,
        num_classes: int = 1,
        device: str = "cpu",
        ffa_mode: bool = False,
    ):
        super().__init__()
        self.radio_version = radio_version
        self.lora_rank = lora_rank
        self.num_lora_layers = num_lora_layers
        self._device = device
        self.ffa_mode = ffa_mode

        # Load shared RADIO backbone
        self.radio = _load_shared_radio(radio_version, device)
        self.backbone = self.radio.model
        self.blocks = self.backbone.blocks
        num_blocks = len(self.blocks)

        # Inject LoRA into last N blocks
        self.lora_layers = []
        start_idx = num_blocks - num_lora_layers
        for i in range(start_idx, num_blocks):
            block = self.blocks[i]
            current_qkv = _get_attn_qkv(block)
            if not isinstance(current_qkv, LoRALinear):
                lora_qkv = LoRALinear(current_qkv, rank=lora_rank, alpha=lora_alpha)
                lora_qkv.to(device)
                _set_attn_qkv(block, lora_qkv)
            else:
                lora_qkv = current_qkv
            self.lora_layers.append((i, lora_qkv))

        # FFA-LoRA: freeze lora_down (A), train only lora_up (B)
        if ffa_mode:
            for _, lora in self.lora_layers:
                lora.lora_down.weight.requires_grad = False

        # Store metadata
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.radio.patch_size

        # Segmentation head
        self.seg_head = SegmentationHead(self.embed_dim, num_classes).to(device)

    @property
    def preferred_resolution(self):
        return self.radio.preferred_resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: images -> RADIO+LoRA -> seg_head -> logits.

        Args:
            x: (B, 3, H, W) images, values in [0, 1].

        Returns:
            (B, num_classes, H, W) segmentation logits.
        """
        img_size = x.shape[-2:]

        # RADIO forward (backbone is frozen except LoRA params)
        output = self.radio(x)
        if isinstance(output, dict):
            _, features = output["backbone"]
        else:
            _, features = output

        # Segmentation head
        logits = self.seg_head(features, img_size)
        return logits

    def get_trainable_params(self):
        """Iterator over all trainable parameters (LoRA + seg_head)."""
        # LoRA params
        for _, lora in self.lora_layers:
            for p in lora.lora_down.parameters():
                if p.requires_grad:
                    yield p
            for p in lora.lora_up.parameters():
                if p.requires_grad:
                    yield p
        # Seg head params
        for p in self.seg_head.parameters():
            if p.requires_grad:
                yield p

    def count_params(self) -> dict:
        """Count trainable and total parameters."""
        lora_down_params = sum(
            p.numel() for _, l in self.lora_layers
            for p in l.lora_down.parameters() if p.requires_grad
        )
        lora_up_params = sum(
            p.numel() for _, l in self.lora_layers
            for p in l.lora_up.parameters() if p.requires_grad
        )
        seg_head_params = sum(
            p.numel() for p in self.seg_head.parameters() if p.requires_grad
        )
        total_trainable = lora_down_params + lora_up_params + seg_head_params
        total_backbone = sum(p.numel() for p in self.radio.parameters())

        return {
            "lora_down": lora_down_params,
            "lora_up": lora_up_params,
            "seg_head": seg_head_params,
            "total_trainable": total_trainable,
            "backbone_frozen": total_backbone,
        }

    def reset_lora(self):
        """Reset LoRA params to initial values."""
        for _, lora in self.lora_layers:
            nn.init.kaiming_uniform_(lora.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(lora.lora_up.weight)
