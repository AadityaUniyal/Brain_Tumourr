"""
Compound Medical Loss Functions:
Combines Cross-Entropy with soft Dice Loss and Focal Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for medical image segmentation."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        # pred_logits: [B, 1, H, W]
        # targets: [B, 1, H, W]
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DualTaskCompoundLoss(nn.Module):
    """
    Multi-task joint loss:
    Loss = alpha * CrossEntropy + beta * DiceLoss + gamma * FocalLoss
    """

    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_fn = DiceLoss()
        self.focal_fn = FocalLoss()

    def forward(self, outputs, batch):
        logits = outputs["logits"]
        seg_mask = outputs["seg_mask"]

        target_label = batch["label"]
        target_mask = batch["mask"]

        # Classification loss
        ce = F.cross_entropy(logits, target_label)
        focal = self.focal_fn(logits, target_label)
        cls_loss = self.ce_weight * ce + self.focal_weight * focal

        # Segmentation loss
        seg_loss = self.dice_fn(seg_mask, target_mask)

        total_loss = cls_loss + self.dice_weight * seg_loss
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "seg_loss": seg_loss.detach(),
        }
