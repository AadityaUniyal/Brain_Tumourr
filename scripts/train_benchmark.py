"""
High-Performance Multi-Task Benchmark Training Engine:
- Optimized for CPU training via 128x128 compact resolution and stage-freezing
- Strict patient-level split (0.0% leakage)
- Joint U-Net segmentation + 3-class subtyping
- Real evaluation on 100% held-out unseen patients
"""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.models.losses import DualTaskCompoundLoss
from src.data.dataset import BrainTumorDataset, get_medical_transforms, LABEL_MAP
from src.data.preprocessor import MedicalImagePreprocessor
from src.training.metrics import compute_clinical_metrics

SPLITS_CSV = "data/splits/patient_stratified_5folds.csv"
OUTPUT_MODEL = "models/checkpoints/neurovision_dualhead_best.pt"
METRICS_OUT = "reports/metrics/benchmark_results.json"
TARGET_SIZE = (128, 128)


def load_compact_dataset(df_subset, prep):
    """Loads and immediately resizes images to compact TARGET_SIZE to conserve RAM."""
    items = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Pre-caching scans"):
        fpath = row["file_path"]
        raw_img, raw_mask, raw_label, pid = prep.load_mat(fpath)

        # Normalize and resize immediately
        norm_img = prep.normalize(raw_img)
        resized_img = cv2.resize(norm_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(raw_mask.astype(np.float32), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        items.append({
            "image": resized_img.astype(np.float32),
            "mask": resized_mask.astype(np.float32),
            "label": LABEL_MAP.get(raw_label, 0),
            "patient_id": pid
        })
    return items


def run_training(epochs=5, batch_size=16, lr=5e-4, val_fold=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Launching High-Performance Multi-Task Benchmark Training on {device} ===")

    df = pd.read_csv(SPLITS_CSV)
    prep = MedicalImagePreprocessor(target_size=TARGET_SIZE)

    train_df = df[(df["split"] != "test") & (df["fold"] != val_fold)]
    val_df = df[(df["split"] != "test") & (df["fold"] == val_fold)]
    test_df = df[df["split"] == "test"]

    print(f"Train Cohort:      {len(train_df)} slices from {train_df['patient_id'].nunique()} patients")
    print(f"Validation Cohort: {len(val_df)} slices from {val_df['patient_id'].nunique()} patients")
    print(f"Test Cohort:        {len(test_df)} slices from {test_df['patient_id'].nunique()} patients")

    # Pre-cache compact datasets
    train_items = load_compact_dataset(train_df, prep)
    val_items = load_compact_dataset(val_df, prep)
    test_items = load_compact_dataset(test_df, prep)

    train_loader = DataLoader(
        BrainTumorDataset(train_items, transforms=get_medical_transforms(is_train=True), target_size=TARGET_SIZE),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        BrainTumorDataset(val_items, transforms=get_medical_transforms(is_train=False), target_size=TARGET_SIZE),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        BrainTumorDataset(test_items, transforms=get_medical_transforms(is_train=False), target_size=TARGET_SIZE),
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model with frozen low-level stages for fast CPU transfer learning
    model = DualHeadBrainTumorNet(num_classes=3, pretrained=True).to(device)

    # Freeze stages 0-2 (ImageNet low-level edge filters)
    for p in list(model.encoder_stage0.parameters()) + list(model.encoder_stage1.parameters()) + list(model.encoder_stage2.parameters()):
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in model.parameters()):,}")

    criterion = DualTaskCompoundLoss(ce_weight=1.0, dice_weight=1.0, focal_weight=0.5)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    history = []

    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, {"label": labels, "mask": masks})

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []
        val_pred_masks, val_true_masks = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss_dict = criterion(outputs, {"label": labels, "mask": masks})
                val_loss += loss_dict["loss"].item()

                probs = torch.softmax(outputs["logits"], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                seg_probs = torch.sigmoid(outputs["seg_mask"]).squeeze(1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_pred_masks.extend(seg_probs.cpu().numpy())
                val_true_masks.extend(masks.squeeze(1).cpu().numpy())

        val_loss /= len(val_loader)
        metrics = compute_clinical_metrics(
            y_true=np.array(val_labels),
            y_pred=np.array(val_preds),
            y_probs=np.array(val_probs),
            pred_masks=np.array(val_pred_masks),
            true_masks=np.array(val_true_masks)
        )

        acc = metrics["overall"]["accuracy"]
        macro_f1 = metrics["overall"]["macro_f1"]
        dice = metrics.get("segmentation", {}).get("mean_dice_coefficient", 0.0)
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {acc:.4f} | Val Macro-F1: {macro_f1:.4f} | Val Dice: {dice:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(acc, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_dice": round(dice, 4)
        })

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  ⭐ Checkpoint saved! New Best Validation Macro-F1: {best_val_f1:.4f}")

    # Final Evaluation on Pure Held-Out Test Set (Patients model has NEVER seen)
    print("\n=== Evaluating Best Model on Pure Held-Out Test Set (Unseen Patients) ===")
    model.load_state_dict(torch.load(OUTPUT_MODEL, map_location=device))
    model.eval()

    test_labels, test_preds, test_probs = [], [], []
    test_pred_masks, test_true_masks = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing on Unseen Patients"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            seg_probs = torch.sigmoid(outputs["seg_mask"]).squeeze(1)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_pred_masks.extend(seg_probs.cpu().numpy())
            test_true_masks.extend(masks.squeeze(1).cpu().numpy())

    test_metrics = compute_clinical_metrics(
        y_true=np.array(test_labels),
        y_pred=np.array(test_preds),
        y_probs=np.array(test_probs),
        pred_masks=np.array(test_pred_masks),
        true_masks=np.array(test_true_masks)
    )

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_fold": val_fold,
        "best_val_macro_f1": round(best_val_f1, 4),
        "held_out_test_results": test_metrics,
        "training_history": history
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print("🏆 FINAL TEST RESULTS ON 100% UNSEEN PATIENT COHORT")
    print("=" * 60)
    print(f"Test Accuracy:         {test_metrics['overall']['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:     {test_metrics['overall']['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:        {test_metrics['overall']['macro_f1']*100:.2f}%")
    print(f"Macro AUROC:           {test_metrics['overall']['auroc_macro']:.4f}")
    print(f"Mean Dice Score (Seg): {test_metrics['segmentation']['mean_dice_coefficient']:.4f}")
    print(f"Expected Calib. Error: {test_metrics['overall']['expected_calibration_error']:.4f}")
    print("=" * 60)
    print(f"Complete benchmark metrics saved to: {METRICS_OUT}")


if __name__ == "__main__":
    run_training(epochs=5, batch_size=16)
