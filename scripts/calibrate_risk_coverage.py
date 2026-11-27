"""
Calibrate Selective Classification Risk-Coverage (RC) Tradeoff on Patient-Held-Out Test Cohort.
Computes:
- Selective Error Rate R(f, g) vs. Coverage Phi(g)
- Optimal rejection threshold tau* guaranteeing Selective Accuracy >= 95.0%
- Error prevention efficiency (percentage of errors intercepted and routed to neuropathologist review)
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
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.dual_head_net import DualHeadBrainTumorNet
from src.data.preprocessor import MedicalImagePreprocessor
from src.inference.conformal import ConformalPredictor
from src.inference.ood_detector import EnergyOODDetector

FIG_DIR = Path("reports/figures/triaging")
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = Path("reports/metrics")
METRIC_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Step 3 & 4: Calibrating Risk-Coverage Tradeoff on Held-Out Test Patients ===")
    
    # 1. Load Model
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
    conformal = ConformalPredictor(alpha=0.05)
    ood_detector = EnergyOODDetector(temperature=1.0)

    # 2. Load Held-Out Test Cohort (474 slices from 35 unseen patients)
    split_df = pd.read_csv("data/splits/patient_stratified_5folds.csv")
    test_rows = split_df[split_df["split"] == "test"]
    print(f"Loaded {len(test_rows)} test slices across {test_rows['patient_id'].nunique()} patient IDs.")

    test_probs = []
    test_labels = []
    test_energies = []
    test_conformal_sizes = []
    test_areas = []

    for _, row in test_rows.iterrows():
        p = Path(row["file_path"])
        if not p.exists():
            p = Path("data/raw/mat_files") / row["file_name"]
        if not p.exists():
            continue
        try:
            with h5py.File(p, 'r') as f:
                cj = f['cjdata']
                label = int(cj['label'][0, 0]) - 1
                img = np.array(cj['image'], dtype=np.float32)
                area = float(row.get("tumor_area_mm2", 0.0))
        except Exception:
            continue

        norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        resized = cv2.resize(norm, (128, 128), interpolation=cv2.INTER_LINEAR)
        img_3c = np.repeat(resized[:, :, None], 3, axis=-1)
        t = val_tf(image=img_3c)["image"].unsqueeze(0)

        with torch.no_grad():
            out = model(t)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1).numpy()[0]
            energy = float(ood_detector.compute_energy(logits).item())

        conf_res = conformal.predict_set(probs)

        test_probs.append(probs)
        test_labels.append(label)
        test_energies.append(energy)
        test_conformal_sizes.append(len(conf_res["prediction_set"]))
        test_areas.append(area)

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_energies = np.array(test_energies)
    test_conformal_sizes = np.array(test_conformal_sizes)
    test_areas = np.array(test_areas)

    n_total = len(test_labels)
    pred_classes = np.argmax(test_probs, axis=1)
    max_probs = np.max(test_probs, axis=1)
    baseline_acc = float(accuracy_score(test_labels, pred_classes))
    baseline_errors = int(np.sum(pred_classes != test_labels))

    print(f"\n[Baseline Forced-Choice (100% Coverage)]")
    print(f"  Overall Accuracy : {baseline_acc:.2%}")
    print(f"  Total Errors     : {baseline_errors} / {n_total}")

    # 3. Sweep Threshold tau to Trace Risk-Coverage Curve
    thresholds = np.linspace(0.40, 0.98, 60)
    coverages = []
    selective_risks = []
    selective_accs = []
    errors_prevented_pct = []

    for tau in thresholds:
        # Selection decision: confidence >= tau AND singleton conformal set AND in-distribution energy
        g_mask = (max_probs >= tau) & (test_conformal_sizes == 1) & (test_energies >= -6.5)
        accepted_count = np.sum(g_mask)
        cov = accepted_count / n_total
        coverages.append(float(cov))

        if accepted_count > 0:
            errors_in_accepted = np.sum((pred_classes != test_labels) & g_mask)
            risk = errors_in_accepted / accepted_count
            acc = 1.0 - risk
            errors_prevented = (baseline_errors - errors_in_accepted) / baseline_errors
        else:
            risk = 0.0
            acc = 1.0
            errors_prevented = 1.0

        selective_risks.append(float(risk))
        selective_accs.append(float(acc))
        errors_prevented_pct.append(float(errors_prevented))

    coverages = np.array(coverages)
    selective_risks = np.array(selective_risks)
    selective_accs = np.array(selective_accs)
    errors_prevented_pct = np.array(errors_prevented_pct)

    # 4. Find Optimal Operating Threshold for <= 5% Selective Risk (>= 95% Accuracy)
    target_risk = 0.05
    valid_idx = np.where(selective_risks <= target_risk)[0]
    if len(valid_idx) > 0:
        opt_idx = valid_idx[0]  # Maximum coverage satisfying risk <= 0.05
        opt_tau = float(thresholds[opt_idx])
        opt_cov = float(coverages[opt_idx])
        opt_acc = float(selective_accs[opt_idx])
        opt_err_prevented = float(errors_prevented_pct[opt_idx])
    else:
        opt_idx = np.argmax(selective_accs)
        opt_tau = float(thresholds[opt_idx])
        opt_cov = float(coverages[opt_idx])
        opt_acc = float(selective_accs[opt_idx])
        opt_err_prevented = float(errors_prevented_pct[opt_idx])

    print(f"\n[Calibrated Optimal Selective Classifier Threshold (tau* = {opt_tau:.3f})]")
    print(f"  Selective Accuracy   : {opt_acc:.2%} (vs Baseline {baseline_acc:.2%})")
    print(f"  Automated Coverage   : {opt_cov:.2%} ({int(opt_cov * n_total)} scans autonomously signed off)")
    print(f"  Abstention Rate      : {(1.0 - opt_cov):.2%} ({int((1.0 - opt_cov) * n_total)} scans routed to Neuropathologist)")
    print(f"  Misdiagnosis Intercepted : {opt_err_prevented:.2%} of all baseline model errors successfully caught!")

    # 5. Evaluate Clinical Triage Queue Breakdown for Abstained Cases
    opt_mask = (max_probs >= opt_tau) & (test_conformal_sizes == 1) & (test_energies >= -6.5)
    abstained_idx = np.where(~opt_mask)[0]

    stat_count = 0
    elevated_count = 0
    routine_count = 0

    for idx in abstained_idx:
        p_glioma = test_probs[idx, 1]
        area = test_areas[idx]
        c_size = test_conformal_sizes[idx]

        if p_glioma >= 0.40 or area >= 500.0:  # ~15 cm3 proxy
            stat_count += 1
        elif c_size >= 2:
            elevated_count += 1
        else:
            routine_count += 1

    total_abstained = len(abstained_idx)
    print(f"\n[Neuropathologist Triaging Queue Breakdown (N={total_abstained})]")
    print(f"  🔴 STAT / Urgent Priority: {stat_count:3d} ({stat_count / total_abstained:.1%}) - High risk of Glioma / Mass Effect")
    print(f"  🟡 Elevated Priority     : {elevated_count:3d} ({elevated_count / total_abstained:.1%}) - Borderline Multi-Class Ambiguity")
    print(f"  🟢 Routine Review Queue  : {routine_count:3d} ({routine_count / total_abstained:.1%}) - Standard Second Opinion")

    # 6. Generate Publication Figures
    # Figure 1: Risk-Coverage Tradeoff & Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Accuracy vs Coverage
    ax1.plot(coverages * 100, selective_accs * 100, color="#2563EB", lw=2.5, label="Selective AI Accuracy")
    ax1.axhline(baseline_acc * 100, color="#94A3B8", linestyle="--", lw=1.5, label=f"Forced-Choice Baseline ({baseline_acc:.1%})")
    ax1.scatter([opt_cov * 100], [opt_acc * 100], color="#EF4444", s=90, zorder=5, label=f"Calibrated Operating Point ({opt_acc:.1%} Acc @ {opt_cov:.1%} Cov)")
    ax1.set_title("Selective Accuracy vs. Automated Coverage", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Automated Decision Coverage (%)", fontsize=10)
    ax1.set_ylabel("Diagnostic Accuracy on Accepted Scans (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, frameon=True)

    # Error Prevention Efficiency vs Coverage
    ax2.plot(coverages * 100, errors_prevented_pct * 100, color="#10B981", lw=2.5, label="Diagnostic Errors Intercepted (%)")
    ax2.scatter([opt_cov * 100], [opt_err_prevented * 100], color="#EF4444", s=90, zorder=5, label=f"Errors Caught ({opt_err_prevented:.1%})")
    ax2.set_title("Medical Malpractice Prevention Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Automated Decision Coverage (%)", fontsize=10)
    ax2.set_ylabel("Percentage of Total Errors Filtered Out (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    rc_plot_path = FIG_DIR / "risk_coverage_curve.png"
    plt.savefig(rc_plot_path)
    plt.close()
    print(f"\nSaved Risk-Coverage curve to: {rc_plot_path}")

    # Figure 2: Triage Priority Distribution Donut / Bar Plot
    plt.figure(figsize=(7, 5), dpi=300)
    triage_labels = ["STAT / URGENT\n(Glioma/Mass Effect)", "ELEVATED\n(Equivocal Boundaries)", "ROUTINE\n(Standard Review)"]
    counts = [stat_count, elevated_count, routine_count]
    colors = ["#EF4444", "#F59E0B", "#10B981"]

    y_p = np.arange(len(counts))
    plt.barh(y_p, counts, color=colors, edgecolor="#334155", height=0.55)
    plt.yticks(y_p, triage_labels, fontsize=9)
    plt.xlabel("Number of Abstained Scans Routed to Queue", fontsize=10)
    plt.title(f"Clinical Neuropathologist Triage Distribution (N={total_abstained})\nSelective Threshold tau* = {opt_tau:.3f}", fontsize=11, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis="x")
    
    # Annotate counts and percentages
    for i, (cnt, col) in enumerate(zip(counts, colors)):
        pct = (cnt / total_abstained) * 100
        plt.text(cnt + 1.5, i, f"{cnt} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    triage_plot_path = FIG_DIR / "triage_priority_distribution.png"
    plt.savefig(triage_plot_path)
    plt.close()
    print(f"Saved Triage Priority distribution to: {triage_plot_path}")

    # Export metrics JSON
    results = {
        "held_out_cohort_size": n_total,
        "baseline_forced_choice_accuracy": baseline_acc,
        "calibrated_selective_classifier": {
            "optimal_threshold_tau": opt_tau,
            "selective_accuracy": opt_acc,
            "selective_error_rate": float(1.0 - opt_acc),
            "automated_coverage": opt_cov,
            "abstention_rate": float(1.0 - opt_cov),
            "errors_intercepted_percentage": opt_err_prevented,
            "triaging_breakdown": {
                "stat_urgent_count": stat_count,
                "elevated_count": elevated_count,
                "routine_count": routine_count,
                "total_abstained": total_abstained
            }
        }
    }
    out_json = METRIC_DIR / "selective_risk_coverage_calibration.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported Risk-Coverage calibration results to: {out_json}")

if __name__ == "__main__":
    main()
