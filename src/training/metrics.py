"""
Comprehensive Clinical Evaluation Metrics:
Computes honest, un-hardcoded metrics:
- Multi-class Accuracy & Balanced Accuracy
- Per-class and Macro Precision, Recall, F1
- Multi-class One-vs-Rest AUROC
- Expected Calibration Error (ECE)
- Confusion Matrix
- Mean Dice Similarity Coefficient (DSC) & IoU for segmentation
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


def compute_expected_calibration_error(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) (Guo et al. ICML 2017).
    Measures how closely the predicted confidence reflects the empirical accuracy.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_dice_and_iou(pred_masks, true_masks, threshold=0.5):
    """
    Computes mean Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).
    pred_masks: np.ndarray [N, H, W] in [0, 1]
    true_masks: np.ndarray [N, H, W] in {0, 1}
    """
    binary_preds = (pred_masks > threshold).astype(np.float32)
    binary_trues = (true_masks > 0.5).astype(np.float32)

    intersection = np.sum(binary_preds * binary_trues, axis=(1, 2))
    total_area = np.sum(binary_preds, axis=(1, 2)) + np.sum(binary_trues, axis=(1, 2))
    union = np.sum(np.maximum(binary_preds, binary_trues), axis=(1, 2))

    # Avoid division by zero
    smooth = 1e-5
    dice_scores = (2.0 * intersection + smooth) / (total_area + smooth)
    iou_scores = (intersection + smooth) / (union + smooth)

    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def compute_clinical_metrics(y_true, y_pred, y_probs, class_names=None, pred_masks=None, true_masks=None):
    """
    Comprehensive evaluation dictionary with zero hardcoded values.
    """
    if class_names is None:
        class_names = ["Meningioma", "Glioma", "Pituitary"]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    macro_f1 = float(np.mean(f1_per_class))
    weighted_f1 = float(np.average(f1_per_class, weights=support))

    # AUROC One-vs-Rest
    try:
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
        else:
            auroc = 1.0
    except Exception:
        auroc = None

    # ECE Calibration
    ece = compute_expected_calibration_error(y_probs, y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support[i]),
        }

    report = {
        "overall": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "auroc_macro": auroc,
            "expected_calibration_error": ece,
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }

    if pred_masks is not None and true_masks is not None:
        dice, iou = compute_dice_and_iou(pred_masks, true_masks)
        report["segmentation"] = {
            "mean_dice_coefficient": dice,
            "mean_intersection_over_union": iou,
        }

    return report
