"""
Post-Hoc Temperature Scaling Calibration (Guo et al., ICML 2017):
Fits scalar temperature T on the validation fold (zero test-set leakage),
saves T into the checkpoint, and updates held_out_test_results in benchmark_results.json.
"""
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.data.dataset import BrainTumorDataset, get_medical_transforms, LABEL_MAP
from src.data.preprocessor import MedicalImagePreprocessor
from src.training.metrics import compute_clinical_metrics

SPLITS_CSV = "data/splits/patient_stratified_5folds.csv"
CKPT_PATH = "models/checkpoints/neurovision_dualhead_best.pt"
METRICS_PATH = "reports/metrics/benchmark_results.json"
TARGET_SIZE = (128, 128)


def load_subset(df_subset, prep):
    items = []
    for _, row in df_subset.iterrows():
        raw_img, raw_mask, raw_label, pid = prep.load_mat(row["file_path"])
        norm_img = prep.normalize(raw_img)
        resized_img = cv2.resize(norm_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(raw_mask.astype(np.float32), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        items.append({
            "image": resized_img.astype(np.float32),
            "mask": resized_mask.astype(np.float32),
            "label": LABEL_MAP.get(raw_label, 0),
            "patient_id": pid,
        })
    return items


def collect_outputs(model, loader, device):
    all_logits, all_labels, all_pred_masks, all_true_masks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["label"].cpu())
            all_pred_masks.extend(torch.sigmoid(out["seg_mask"]).squeeze(1).cpu().numpy())
            all_true_masks.extend(batch["mask"].squeeze(1).cpu().numpy())
    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        np.array(all_pred_masks),
        np.array(all_true_masks),
    )


def main():
    device = "cpu"
    df = pd.read_csv(SPLITS_CSV)
    prep = MedicalImagePreprocessor(target_size=TARGET_SIZE)
    val_df = df[(df["split"] != "test") & (df["fold"] == 0)]
    test_df = df[df["split"] == "test"]

    val_loader = DataLoader(BrainTumorDataset(load_subset(val_df, prep), transforms=get_medical_transforms(False), target_size=TARGET_SIZE), batch_size=32, shuffle=False)
    test_loader = DataLoader(BrainTumorDataset(load_subset(test_df, prep), transforms=get_medical_transforms(False), target_size=TARGET_SIZE), batch_size=32, shuffle=False)

    model = DualHeadBrainTumorNet(num_classes=3, pretrained=False)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()

    val_logits, val_labels, _, _ = collect_outputs(model, val_loader, device)
    test_logits, test_labels, test_pred_masks, test_true_masks = collect_outputs(model, test_loader, device)

    # Fit temperature T on validation fold NLL
    best_T, best_nll = 1.0, float("inf")
    for t in np.linspace(0.3, 2.0, 171):
        nll = F.cross_entropy(val_logits / float(t), val_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_T = float(t)

    print(f"Optimal Validation Temperature T = {best_T:.3f} (Val NLL = {best_nll:.4f})")

    probs = torch.softmax(test_logits / best_T, dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    test_metrics = compute_clinical_metrics(
        y_true=test_labels.numpy(),
        y_pred=preds,
        y_probs=probs,
        pred_masks=test_pred_masks,
        true_masks=test_true_masks,
    )
    print(f"Calibrated Test ECE: {test_metrics['overall']['expected_calibration_error']:.4f} | AUROC: {test_metrics['overall']['auroc_macro']:.4f} | Dice: {test_metrics['segmentation']['mean_dice_coefficient']:.4f}")

    if isinstance(ckpt, dict):
        ckpt["temperature"] = best_T
        torch.save(ckpt, CKPT_PATH)

    with open(METRICS_PATH) as f:
        report = json.load(f)
    report["temperature_scaling_T"] = round(best_T, 4)
    report["held_out_test_results"] = test_metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
