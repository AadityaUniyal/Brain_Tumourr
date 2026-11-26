"""
Tests for OOD Safety Detector and Conformal Prediction.
"""
import torch
import numpy as np
from src.inference.ood_detector import EnergyOODDetector
from src.inference.conformal import ConformalPredictor


def test_energy_ood_detector():
    detector = EnergyOODDetector(energy_threshold=-5.0, temperature=1.0)
    
    # Sharp in-distribution logits
    in_dist_logits = torch.tensor([[10.0, 1.0, 0.5]])
    is_ood_in, energy_in, _ = detector.check(in_dist_logits)
    assert not is_ood_in
    assert energy_in < -5.0

    # Diffuse out-of-distribution logits (e.g. random noise or non-brain scan)
    ood_logits = torch.tensor([[0.1, 0.1, 0.1]])
    is_ood_out, energy_out, _ = detector.check(ood_logits)
    assert is_ood_out
    assert energy_out > -5.0


def test_conformal_predictor():
    cp = ConformalPredictor(alpha=0.05, class_names=["Meningioma", "Glioma", "Pituitary"])
    
    # Simulated calibration set
    calib_probs = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.85, 0.1],
        [0.02, 0.08, 0.9],
        [0.8, 0.1, 0.1]
    ])
    calib_labels = np.array([0, 1, 2, 0])
    q_hat = cp.calibrate(calib_probs, calib_labels)
    assert 0.0 <= q_hat <= 1.0

    # Confident test prediction
    res_confident = cp.predict_set(np.array([0.95, 0.03, 0.02]))
    assert "Meningioma" in res_confident["prediction_set"]

    # Ambiguous borderline test prediction
    res_ambiguous = cp.predict_set(np.array([0.48, 0.48, 0.04]))
    assert res_ambiguous["is_uncertain"]
