"""
Tests for DualHeadBrainTumorNet architecture and Compound Losses.
"""
import torch
import pytest
from src.models.dual_head_net import DualHeadBrainTumorNet
from src.models.losses import DualTaskCompoundLoss


def test_dual_head_forward_pass():
    model = DualHeadBrainTumorNet(num_classes=3, pretrained=False)
    model.eval()

    # Synthetic batch of 2 images: [Batch, Channels, Height, Width]
    x = torch.randn(2, 3, 128, 128)
    outputs = model(x)

    assert "logits" in outputs
    assert "seg_mask" in outputs
    assert outputs["logits"].shape == (2, 3), f"Expected logits shape (2, 3), got {outputs['logits'].shape}"
    assert outputs["seg_mask"].shape == (2, 1, 128, 128), f"Expected mask shape (2, 1, 128, 128), got {outputs['seg_mask'].shape}"


def test_compound_loss_backward():
    model = DualHeadBrainTumorNet(num_classes=3, pretrained=False)
    criterion = DualTaskCompoundLoss(ce_weight=1.0, dice_weight=1.0, focal_weight=0.5)

    x = torch.randn(2, 3, 64, 64)
    target_labels = torch.tensor([0, 2], dtype=torch.long)
    target_masks = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    target_masks[0, 0, 10:30, 10:30] = 1.0

    outputs = model(x)
    loss_dict = criterion(outputs, {"label": target_labels, "mask": target_masks})

    total_loss = loss_dict["loss"]
    assert total_loss.item() > 0.0

    total_loss.backward()
    # Verify gradients computed
    assert model.classifier[-1].weight.grad is not None
