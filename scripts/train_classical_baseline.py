"""
Train classical clinical and radiomic baselines for:
1. Overall Survival (Cox Proportional Hazards via lifelines) -> Harrell's C-index
2. Molecular Markers (IDH Mutation & 1p/19q Codeletion via Random Forest / Logistic Regression) -> AUROC & F1
Establishes the empirical benchmark before deep multimodal modeling.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

DATA_PATH = Path("data/processed/tcga_clinical_molecular_survival.csv")
OUT_METRICS = Path("reports/metrics/classical_baseline_results.json")
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Training Classical Clinical & Radiomic Baselines ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} validated patient records.")

    # Prepare features
    # Numerical: Age
    # Categorical: Sex (M/F), Cohort (LGG/GBM)
    df["is_male"] = (df["sex"] == "MALE").astype(int)
    df["is_gbm"] = (df["cohort"] == "GBM").astype(int)

    # 1. Classical Survival Baseline: Cox Proportional Hazards
    print("\n--- 1. Classical Survival Model: Cox Proportional Hazards ---")
    cph_df = df[["os_months", "vital_status", "age", "is_male", "is_gbm"]].copy()
    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="os_months", event_col="vital_status")
    cph_summary = cph.summary[["coef", "exp(coef)", "p"]]
    print(cph_summary)

    # 5-Fold Cross-Validated C-index for Cox-PH
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []
    for train_idx, val_idx in skf.split(cph_df, cph_df["vital_status"]):
        train_data = cph_df.iloc[train_idx]
        val_data = cph_df.iloc[val_idx]
        cph_fold = CoxPHFitter()
        cph_fold.fit(train_data, duration_col="os_months", event_col="vital_status")
        # Predict partial hazards on validation set
        val_hazards = cph_fold.predict_partial_hazard(val_data)
        # Note: higher hazard means shorter survival, so we negate for concordance with duration
        c_val = concordance_index(val_data["os_months"], -val_hazards, val_data["vital_status"])
        c_indices.append(c_val)

    mean_c_index = float(np.mean(c_indices))
    std_c_index = float(np.std(c_indices))
    print(f"\nClassical Clinical Cox-PH Mean C-index across 5 Folds: {mean_c_index:.4f} ± {std_c_index:.4f}")

    # 2. Molecular Baselines: IDH Mutation & 1p/19q Codeletion
    print("\n--- 2. Molecular Target Baselines: Random Forest & Logistic Regression ---")
    X = df[["age", "is_male", "is_gbm"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    targets = {
        "idh_mutation": df["idh_mutation"].values,
        "codeletion_1p19q": df["codeletion_1p19q"].values
    }

    baseline_results = {
        "survival_coxph": {
            "mean_c_index": mean_c_index,
            "std_c_index": std_c_index
        },
        "molecular_baselines": {}
    }

    for target_name, y in targets.items():
        rf_aucs, rf_f1s = [], []
        lr_aucs, lr_f1s = [], []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, y_tr = X_scaled[train_idx], y[train_idx]
            X_va, y_val = X_scaled[val_idx], y[val_idx]

            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight="balanced")
            rf.fit(X_tr, y_tr)
            y_pred_prob_rf = rf.predict_proba(X_va)[:, 1]
            y_pred_rf = (y_pred_prob_rf >= 0.5).astype(int)
            rf_aucs.append(roc_auc_score(y_val, y_pred_prob_rf))
            rf_f1s.append(f1_score(y_val, y_pred_rf, zero_division=0))

            # Logistic Regression
            lr = LogisticRegression(class_weight="balanced", random_state=42)
            lr.fit(X_tr, y_tr)
            y_pred_prob_lr = lr.predict_proba(X_va)[:, 1]
            y_pred_lr = (y_pred_prob_lr >= 0.5).astype(int)
            lr_aucs.append(roc_auc_score(y_val, y_pred_prob_lr))
            lr_f1s.append(f1_score(y_val, y_pred_lr, zero_division=0))

        print(f"\nTarget: {target_name}")
        print(f"  Random Forest AUROC: {np.mean(rf_aucs):.4f} ± {np.std(rf_aucs):.4f} | F1: {np.mean(rf_f1s):.4f}")
        print(f"  Logistic Reg  AUROC: {np.mean(lr_aucs):.4f} ± {np.std(lr_aucs):.4f} | F1: {np.mean(lr_f1s):.4f}")

        baseline_results["molecular_baselines"][target_name] = {
            "rf_mean_auroc": float(np.mean(rf_aucs)),
            "rf_mean_f1": float(np.mean(rf_f1s)),
            "lr_mean_auroc": float(np.mean(lr_aucs)),
            "lr_mean_f1": float(np.mean(lr_f1s))
        }

    with open(OUT_METRICS, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\nSaved baseline benchmark results to: {OUT_METRICS}")

if __name__ == "__main__":
    main()
