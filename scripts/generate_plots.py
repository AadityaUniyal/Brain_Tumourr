"""
Generates publication-quality figures from benchmark results:
1. reports/figures/training_curves.png
2. reports/figures/confusion_matrix.png
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

METRICS_JSON = "reports/metrics/benchmark_results.json"
FIGS_DIR = "reports/figures"
CLASS_NAMES = ["Meningioma", "Glioma", "Pituitary"]


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    with open(METRICS_JSON, "r") as f:
        data = json.load(f)

    history = data["training_history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] * 100 for h in history]
    val_f1 = [h["val_macro_f1"] * 100 for h in history]
    val_dice = [h["val_dice"] for h in history]

    # 1. Training & Validation Curves
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curves
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#e74c3c", linewidth=2)
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="#2980b9", linewidth=2)
    axes[0].set_title("Multi-Task Compound Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy & Macro-F1
    axes[1].plot(epochs, val_acc, "o-", label="Val Accuracy (%)", color="#27ae60", linewidth=2)
    axes[1].plot(epochs, val_f1, "s--", label="Val Macro-F1 (%)", color="#8e44ad", linewidth=2)
    axes[1].set_title("Validation Classification Metrics", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend()

    # Dice Segmentation Score
    axes[2].plot(epochs, val_dice, "d-", label="Mean Dice Score", color="#d35400", linewidth=2)
    axes[2].set_title("U-Net Tumor Segmentation Overlap", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice Similarity Coefficient")
    axes[2].legend()

    plt.suptitle("NeuroVision Elite Multi-Task Training History (Zero Patient Leakage)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    curves_path = os.path.join(FIGS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"✅ Training curves saved to: {curves_path}")

    # 2. Confusion Matrix Heatmap
    cm = np.array(data["held_out_test_results"]["confusion_matrix"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
    )
    plt.title(f"Test Confusion Matrix on 35 Unseen Patients\n(Overall Accuracy: {data['held_out_test_results']['overall']['accuracy']*100:.2f}%)", fontsize=12, fontweight="bold")
    plt.ylabel("True Pathological Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_path = os.path.join(FIGS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
