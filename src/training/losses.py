"""
Loss functions for segmentation.

Includes:
- Binary Cross-Entropy (BCE)
- Dice Loss
- Combined Dice + BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    """Wrapper for BCE with logits."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss.

    This is often better than either alone:
    - BCE provides pixel-wise gradients (good for learning)
    - Dice provides region-level gradients (good for class imbalance)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def create_loss(
    name: str = "dice_bce",
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
) -> nn.Module:
    """
    Factory function to create loss.

    Args:
        name: "bce", "dice", or "dice_bce"
        dice_weight: Weight for dice loss (only for "dice_bce")
        bce_weight: Weight for BCE loss (only for "dice_bce")

    Returns:
        Loss module
    """
    if name == "bce":
        return BCEWithLogitsLoss()
    elif name == "dice":
        return DiceLoss()
    elif name == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
    else:
        raise ValueError(f"Unknown loss: {name}")
