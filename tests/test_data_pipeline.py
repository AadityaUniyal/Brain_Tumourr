"""
Tests for Data Preprocessing and Patient-Grouped Splitting.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import MedicalImagePreprocessor
from src.data.split import generate_patient_stratified_splits, audit_split_integrity


def test_preprocessor_shape_and_normalization():
    prep = MedicalImagePreprocessor(target_size=(128, 128))
    # Create synthetic float image
    synthetic_img = np.random.uniform(50, 200, (200, 200)).astype(np.float32)
    norm = prep.normalize(synthetic_img)
    assert np.min(norm) >= 0.0
    assert np.max(norm) <= 1.0


def test_zero_patient_leakage_split():
    # Construct mock dataset across 10 patients with multiple slices per patient
    records = []
    for pid in range(1, 11):
        num_slices = np.random.randint(3, 8)
        label = pid % 3
        for s in range(num_slices):
            records.append({
                "file_path": f"scan_p{pid}_s{s}.jpg",
                "patient_id": f"PATIENT_{pid:03d}",
                "label": label
            })

    df_splits = generate_patient_stratified_splits(records, n_splits=3, test_size=0.2, random_state=42)

    # Assert test split does not leak patients into train
    train_patients = set(df_splits[df_splits["split"] == "train"]["patient_id"])
    test_patients = set(df_splits[df_splits["split"] == "test"]["patient_id"])
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage found in test set!"

    # Audit integrity
    report = audit_split_integrity(df_splits)
    assert report["test_patients"] > 0
    for fold_name, fold_data in report["folds"].items():
        assert fold_data["patient_leakage_count"] == 0, f"Patient leakage found in {fold_name}!"
