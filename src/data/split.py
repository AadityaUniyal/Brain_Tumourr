"""
Zero Patient Leakage Split Engine:
Implements StratifiedGroupKFold cross-validation partitioning.
Guarantees that 100% of an individual patient's MRI slices are assigned exclusively
to either Train, Validation, or Test - never leaked across splits.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def generate_patient_stratified_splits(metadata_records, n_splits=5, test_size=0.15, random_state=42):
    """
    Takes a list of scan metadata dicts:
        [{'file_path': '...', 'patient_id': 'PID_001', 'label': 1}, ...]
    
    Returns:
        df_manifest (pd.DataFrame): Annotated with 'fold' and 'split' ('train', 'val', 'test')
    """
    df = pd.DataFrame(metadata_records)
    if "patient_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Metadata records must contain 'patient_id' and 'label' columns.")

    # Step 1: Hold out a clean, pure test set grouped strictly by patient_id
    unique_patients = df.drop_duplicates(subset=["patient_id"])[["patient_id", "label"]]
    
    n_unique = len(unique_patients)
    n_classes = unique_patients["label"].nunique()
    min_class_count = unique_patients["label"].value_counts().min() if n_classes > 0 else 0
    calculated_test_n = int(np.ceil(n_unique * test_size)) if isinstance(test_size, float) else test_size

    # Check if stratified split is mathematically possible
    can_stratify = (min_class_count >= 2) and (calculated_test_n >= n_classes) and ((n_unique - calculated_test_n) >= n_classes)

    train_val_patients, test_patients = train_test_split(
        unique_patients["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=unique_patients["label"] if can_stratify else None,
    )

    df["split"] = "train"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

    # Step 2: On the train+val subset, apply StratifiedGroupKFold
    train_val_df = df[df["split"] != "test"].copy().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_val_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf.split(
            X=train_val_df["file_path"],
            y=train_val_df["label"],
            groups=train_val_df["patient_id"],
        )
    ):
        train_val_df.loc[val_idx, "fold"] = fold

    # Map back to main DataFrame
    df["fold"] = -1
    for idx, row in train_val_df.iterrows():
        orig_match = (df["file_path"] == row["file_path"]) & (df["patient_id"] == row["patient_id"])
        df.loc[orig_match, "fold"] = row["fold"]

    # Audit for zero leakage
    train_pids = set(df[df["split"] != "test"]["patient_id"])
    test_pids = set(df[df["split"] == "test"]["patient_id"])
    leakage = train_pids.intersection(test_pids)
    if leakage:
        raise RuntimeError(f"CRITICAL ERROR: Patient leakage detected across test split! Patients: {leakage}")

    return df


def audit_split_integrity(df_manifest):
    """
    Computes and logs patient overlap statistics across all folds.
    """
    report = {
        "total_slices": len(df_manifest),
        "total_unique_patients": df_manifest["patient_id"].nunique(),
        "test_slices": len(df_manifest[df_manifest["split"] == "test"]),
        "test_patients": df_manifest[df_manifest["split"] == "test"]["patient_id"].nunique(),
        "folds": {},
    }

    for fold in sorted(df_manifest["fold"].unique()):
        if fold == -1:
            continue
        val_slice_count = len(df_manifest[df_manifest["fold"] == fold])
        val_pids = set(df_manifest[df_manifest["fold"] == fold]["patient_id"])
        train_pids = set(df_manifest[(df_manifest["fold"] != fold) & (df_manifest["split"] != "test")]["patient_id"])
        overlap = len(val_pids.intersection(train_pids))
        report["folds"][f"fold_{fold}"] = {
            "val_slices": val_slice_count,
            "val_patients": len(val_pids),
            "patient_leakage_count": overlap,
        }
    return report
