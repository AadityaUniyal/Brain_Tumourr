"""
Compound Medical Loss Functions:
Combines Label-Smoothed Cross-Entropy, Focal Loss, and BCE + Per-Sample Focal Tversky & Dice Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Per-sample Soft Dice Loss so small lesions (Pituitary/Meningioma) are weighted equally."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        probs = torch.sigmoid(pred_logits)
        B = probs.shape[0]
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced medical image segmentation (Salehi et al., 2017).
    alpha=0.7 penalizes false negatives (under-segmentation of diffuse margins)
    beta=0.3 penalizes false positives
    gamma=1.33 focuses on hard ambiguous boundary voxels
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        probs = torch.sigmoid(pred_logits)
        B = probs.shape[0]
        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)
        return focal_tversky.mean()


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
    Loss = cls_loss + dice_weight * (0.35 * BCE + 0.35 * SoftDice + 0.30 * FocalTversky)
    """

    def __init__(self, ce_weight=1.0, dice_weight=2.5, focal_weight=0.5, label_smoothing=0.05):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
        self.dice_fn = DiceLoss()
        self.tversky_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.focal_fn = FocalLoss()

    def forward(self, outputs, batch):
        logits = outputs["logits"]
        seg_mask = outputs["seg_mask"]

        target_label = batch["label"]
        target_mask = batch["mask"]

        ce = F.cross_entropy(logits, target_label, label_smoothing=self.label_smoothing)
        focal = self.focal_fn(logits, target_label)
        cls_loss = self.ce_weight * ce + self.focal_weight * focal

        seg_dice = self.dice_fn(seg_mask, target_mask)
        seg_bce = self.bce_fn(seg_mask, target_mask)
        seg_tversky = self.tversky_fn(seg_mask, target_mask)
        seg_loss = 0.35 * seg_dice + 0.35 * seg_bce + 0.30 * seg_tversky

        total_loss = cls_loss + self.dice_weight * seg_loss
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "seg_loss": seg_loss.detach(),
        }
