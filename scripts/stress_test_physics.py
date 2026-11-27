"""
Physics-Based Scanner Inhomogeneity and Artifact Stress Testing for Brain MRI AI.
Simulates:
1. B1 Magnetic Field Inhomogeneity (Smooth Multiplicative Bias Field Drift)
2. Rician Noise Perturbation (Low Signal-to-Noise Ratio Magnitude MRI)
3. Gibbs Ringing & Motion Artifacts (Fourier Domain k-space Truncation & Phase Modulation)
Evaluates robustness bounds and degradation curves across artifact severities.
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
import cv2
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.data.preprocessor import MedicalImagePreprocessor

FIG_DIR = Path("reports/figures/radiogenomics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = Path("reports/metrics")
METRIC_DIR.mkdir(parents=True, exist_ok=True)

# 1. Physical Distortion Functions
def apply_b1_bias_field(image: np.ndarray, severity: float = 0.3) -> np.ndarray:
    """Simulates RF coil sensitivity B1 drift via 2D spatial quadratic field."""
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    # Normalized radial and linear coordinates
    nx = (x - cx) / (w / 2.0)
    ny = (y - cy) / (h / 2.0)
    # Smooth quadratic field
    bias = 1.0 + severity * (0.5 * nx + 0.3 * ny - 0.4 * (nx**2 + ny**2))
    distorted = image * bias
    return np.clip(distorted, 0.0, None)

def apply_rician_noise(image: np.ndarray, snr_sigma: float = 0.08) -> np.ndarray:
    """Simulates MRI magnitude image Rician noise distribution."""
    # Scale max intensity to 1.0 for noise calculation
    peak = image.max() if image.max() > 0 else 1.0
    norm_img = image / peak
    
    n_real = np.random.normal(0, snr_sigma, size=image.shape)
    n_imag = np.random.normal(0, snr_sigma, size=image.shape)
    
    rician = np.sqrt((norm_img + n_real) ** 2 + n_imag ** 2)
    return rician * peak

def apply_gibbs_ringing_and_motion(image: np.ndarray, cutoff_ratio: float = 0.65, phase_drift: float = 0.3) -> np.ndarray:
    """Simulates k-space truncation (Gibbs ringing) and phase modulation motion."""
    # 2D Fourier Transform
    f_space = np.fft.fftshift(np.fft.fft2(image))
    h, w = image.shape
    cy, cx = h // 2, w // 2
    
    # Low-pass rectangular filter (truncating high k-space frequencies)
    mask = np.zeros_like(f_space, dtype=float)
    ky_cut = int(cy * cutoff_ratio)
    kx_cut = int(cx * cutoff_ratio)
    mask[cy - ky_cut:cy + ky_cut, cx - kx_cut:cx + kx_cut] = 1.0
    
    # Phase disturbance across phase-encoding direction (y-axis)
    phase_jitter = np.exp(1j * phase_drift * np.sin(np.linspace(0, 4 * np.pi, h)))[:, None]
    
    f_space_corrupted = f_space * mask * phase_jitter
    corrupted_img = np.abs(np.fft.ifft2(np.fft.ifftshift(f_space_corrupted)))
    return corrupted_img

def main():
    print("=== Step 3 & 4: Executing MRI Physics Distortion Stress Test ===")
    
    # Load Benchmark Model
    model = DualHeadBrainTumorNet(num_classes=3)
    weights_path = Path("models/checkpoints/neurovision_dualhead_best.pt")
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=torch.device("cpu"))
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    model.eval()

    from src.data.dataset import get_medical_transforms
    val_tf = get_medical_transforms(is_train=False)

    # Load 100 held-out test scans from split
    split_df = pd.read_csv("data/splits/patient_stratified_5folds.csv")
    test_rows = split_df[split_df["split"] == "test"].head(100)
    print(f"Testing across {len(test_rows)} patient-isolated test scans...")

    preprocessor = MedicalImagePreprocessor(target_size=(128, 128))

    # Load clean samples
    clean_images = []
    clean_labels = []
    clean_masks = []
    
    for _, row in test_rows.iterrows():
        p = Path(row["file_path"])
        if not p.exists():
            # Fallback if path separator
            p = Path("data/raw/mat_files") / row["file_name"]
        if not p.exists():
            continue
        try:
            with h5py.File(p, 'r') as f:
                cj = f['cjdata']
                label = int(cj['label'][0, 0]) - 1 # 0-indexed
                img = np.array(cj['image'], dtype=np.float32)
                mask = np.array(cj['tumorMask'], dtype=np.uint8)
                clean_images.append(img)
                clean_labels.append(label)
                clean_masks.append(mask)
        except Exception:
            continue

    clean_labels = np.array(clean_labels)
    print(f"Loaded {len(clean_images)} verified evaluation scans.")

    # Generate Distortion Gallery Visualization
    sample_img = clean_images[0]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=300)
    axes[0].imshow(sample_img, cmap="bone")
    axes[0].set_title("1. Baseline Pristine MRI", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    b1_img = apply_b1_bias_field(sample_img, severity=0.35)
    axes[1].imshow(b1_img, cmap="bone")
    axes[1].set_title("2. B1 Field Inhomogeneity\n(Coil Sensitivity Drift)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    rician_img = apply_rician_noise(sample_img, snr_sigma=0.10)
    axes[2].imshow(rician_img, cmap="bone")
    axes[2].set_title("3. Rician Noise Perturbation\n(Low SNR 1.5T Scanner)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    gibbs_img = apply_gibbs_ringing_and_motion(sample_img, cutoff_ratio=0.6, phase_drift=0.4)
    axes[3].imshow(gibbs_img, cmap="bone")
    axes[3].set_title("4. Gibbs Ringing & Motion\n(k-Space Phase Artifact)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    gallery_path = FIG_DIR / "physics_stress_test_gallery.png"
    plt.savefig(gallery_path)
    plt.close()
    print(f"Saved distortion gallery to: {gallery_path}")

    # Evaluate Degradation Across Perturbation Severities
    def evaluate_cohort(img_list):
        preds = []
        for img in img_list:
            norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
            resized = cv2.resize(norm, (128, 128), interpolation=cv2.INTER_LINEAR)
            img_3c = np.repeat(resized[:, :, None], 3, axis=-1)
            t = val_tf(image=img_3c)["image"].unsqueeze(0)
            with torch.no_grad():
                out = model(t)
                logits = out["logits"]
                probs = torch.softmax(logits, dim=1).numpy()[0]
                preds.append(probs)
        preds = np.array(preds)
        pred_cls = np.argmax(preds, axis=1)
        acc = accuracy_score(clean_labels, pred_cls)
        try:
            auc = roc_auc_score(clean_labels, preds, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.5
        return float(acc), float(auc)

    # 1. Baseline
    base_acc, base_auc = evaluate_cohort(clean_images)
    print(f"\n[Baseline Pristine]  Accuracy: {base_acc:.2%} | Macro AUROC: {base_auc:.4f}")

    # 2. B1 Bias Field levels
    b1_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    b1_aucs, b1_accs = [], []
    for s in b1_levels:
        dist_imgs = [apply_b1_bias_field(img, severity=s) for img in clean_images]
        acc, auc = evaluate_cohort(dist_imgs)
        b1_accs.append(acc)
        b1_aucs.append(auc)
        print(f"  B1 Bias (severity={s:.1f}) -> Accuracy: {acc:.2%} | AUROC: {auc:.4f}")

    # 3. Rician Noise levels
    rician_levels = [0.03, 0.06, 0.09, 0.12, 0.15]
    ric_aucs, ric_accs = [], []
    for s in rician_levels:
        dist_imgs = [apply_rician_noise(img, snr_sigma=s) for img in clean_images]
        acc, auc = evaluate_cohort(dist_imgs)
        ric_accs.append(acc)
        ric_aucs.append(auc)
        print(f"  Rician Noise (sigma={s:.2f}) -> Accuracy: {acc:.2%} | AUROC: {auc:.4f}")

    # 4. Gibbs Ringing & Motion levels
    motion_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    mot_aucs, mot_accs = [], []
    for m in motion_levels:
        dist_imgs = [apply_gibbs_ringing_and_motion(img, cutoff_ratio=0.7 - m*0.3, phase_drift=m) for img in clean_images]
        acc, auc = evaluate_cohort(dist_imgs)
        mot_accs.append(acc)
        mot_aucs.append(auc)
        print(f"  Motion/Gibbs (drift={m:.1f}) -> Accuracy: {acc:.2%} | AUROC: {auc:.4f}")

    # Plot Degradation Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    # Accuracy curves
    severities = [1, 2, 3, 4, 5]
    ax1.plot(severities, b1_accs, marker="o", color="#3B82F6", lw=2, label="B1 Bias Field Inhomogeneity")
    ax1.plot(severities, ric_accs, marker="s", color="#EF4444", lw=2, label="Rician Noise (Low SNR)")
    ax1.plot(severities, mot_accs, marker="^", color="#F59E0B", lw=2, label="Gibbs Ringing & Motion")
    ax1.axhline(base_acc, color="#10B981", linestyle="--", lw=1.5, label=f"Clean Baseline ({base_acc:.1%})")
    ax1.set_title("Accuracy vs. Distortion Severity", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Distortion Severity Level (1 = Mild, 5 = Severe)", fontsize=10)
    ax1.set_ylabel("Diagnostic Accuracy", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8, frameon=True)

    # AUROC curves
    ax2.plot(severities, b1_aucs, marker="o", color="#3B82F6", lw=2, label="B1 Bias Field")
    ax2.plot(severities, ric_aucs, marker="s", color="#EF4444", lw=2, label="Rician Noise")
    ax2.plot(severities, mot_aucs, marker="^", color="#F59E0B", lw=2, label="Motion / Gibbs Ringing")
    ax2.axhline(base_auc, color="#10B981", linestyle="--", lw=1.5, label=f"Clean Baseline ({base_auc:.3f})")
    ax2.set_title("Macro AUROC vs. Distortion Severity", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Distortion Severity Level (1 = Mild, 5 = Severe)", fontsize=10)
    ax2.set_ylabel("Macro AUROC", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    curve_path = FIG_DIR / "robustness_degradation_curves.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved degradation curves to: {curve_path}")

    # Export metrics JSON
    results = {
        "baseline": {
            "accuracy": base_acc,
            "macro_auroc": base_auc
        },
        "stress_tests": {
            "b1_bias_field": {
                "severities": b1_levels,
                "accuracy": b1_accs,
                "macro_auroc": b1_aucs,
                "max_auroc_drop": float(base_auc - min(b1_aucs))
            },
            "rician_noise": {
                "sigmas": rician_levels,
                "accuracy": ric_accs,
                "macro_auroc": ric_aucs,
                "max_auroc_drop": float(base_auc - min(ric_aucs))
            },
            "gibbs_motion": {
                "phase_drifts": motion_levels,
                "accuracy": mot_accs,
                "macro_auroc": mot_aucs,
                "max_auroc_drop": float(base_auc - min(mot_aucs))
            }
        }
    }
    out_json = METRIC_DIR / "physics_robustness_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported physics robustness metrics to: {out_json}")

if __name__ == "__main__":
    main()
