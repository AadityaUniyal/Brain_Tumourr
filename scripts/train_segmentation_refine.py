"""
Segmentation Refinement & Fine-Tuning Engine:
Resumes from the best checkpoint (Macro AUROC 0.9737) and fine-tunes using
the new Focal-Tversky + BCE + SoftDice compound loss to lift Mean Dice score.
Enforces zero patient data leakage.
"""
import os
import sys
import json
import time
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.models.losses import DualTaskCompoundLoss
from src.data.dataset import BrainTumorDataset, get_medical_transforms, LABEL_MAP
from src.data.preprocessor import MedicalImagePreprocessor
from src.training.metrics import compute_clinical_metrics

SPLITS_CSV = "data/splits/patient_stratified_5folds.csv"
INPUT_MODEL = "models/checkpoints/neurovision_dualhead_best.pt"
OUTPUT_MODEL = "models/checkpoints/neurovision_dualhead_best.pt"
METRICS_OUT = "reports/metrics/benchmark_results.json"
TARGET_SIZE = (128, 128)


def load_compact_dataset(df_subset, prep):
    items = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Pre-caching scans"):
        fpath = row["file_path"]
        raw_img, raw_mask, raw_label, pid = prep.load_mat(fpath)
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


def run_refinement(epochs=6, batch_size=16, lr=2.5e-4, val_fold=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"\n=== Launching Segmentation Refinement on {device} ({torch.get_num_threads()} threads) ===")

    df = pd.read_csv(SPLITS_CSV)
    prep = MedicalImagePreprocessor(target_size=TARGET_SIZE)

    train_df = df[(df["split"] != "test") & (df["fold"] != val_fold)]
    val_df = df[(df["split"] != "test") & (df["fold"] == val_fold)]
    test_df = df[df["split"] == "test"]

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

    model = DualHeadBrainTumorNet(num_classes=3, pretrained=False).to(device)
    if os.path.exists(INPUT_MODEL):
        ckpt = torch.load(INPUT_MODEL, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
        print(f"Loaded existing checkpoint from {INPUT_MODEL}")

    # Differential learning rate: decoder & seg_head get full LR, encoder gets small refinement LR
    encoder_params = [p for n, p in model.named_parameters() if "encoder_" in n]
    decoder_params = [p for n, p in model.named_parameters() if "encoder_" not in n]

    criterion = DualTaskCompoundLoss(ce_weight=0.8, dice_weight=3.0, focal_weight=0.4, label_smoothing=0.04)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr * 0.2},
        {"params": decoder_params, "lr": lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_joint = -1.0
    best_val_f1 = -1.0
    best_val_dice = -1.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Refine Epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, {"label": labels, "mask": masks})

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        joint = 0.40 * macro_f1 + 0.60 * dice

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {macro_f1:.4f} | Dice: {dice:.4f} | Joint: {joint:.4f}")

        if joint > best_joint:
            best_joint = joint
            best_val_f1 = macro_f1
            best_val_dice = dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_macro_f1": round(macro_f1, 4),
                "val_dice": round(dice, 4),
                "target_size": TARGET_SIZE,
            }, OUTPUT_MODEL)
            print(f"  ⭐ Checkpoint saved! New Best Joint: {best_joint:.4f} (Dice: {dice:.4f}, F1: {macro_f1:.4f})")

    # Evaluate on held-out test cohort
    print("\n=== Evaluating Refined Model on Held-Out Test Set ===")
    ckpt = torch.load(OUTPUT_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
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
        "best_val_dice": round(best_val_dice, 4),
        "held_out_test_results": test_metrics,
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 60)
    print("🏆 REFINED TEST RESULTS ON UNSEEN PATIENT COHORT")
    print("=" * 60)
    print(f"Test Accuracy:         {test_metrics['overall']['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:     {test_metrics['overall']['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:        {test_metrics['overall']['macro_f1']*100:.2f}%")
    print(f"Macro AUROC:           {test_metrics['overall']['auroc_macro']:.4f}")
    print(f"Mean Dice Score (Seg): {test_metrics['segmentation']['mean_dice_coefficient']:.4f}")
    print(f"Expected Calib. Error: {test_metrics['overall']['expected_calibration_error']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_refinement(epochs=6, batch_size=16, lr=2.5e-4)
