"""
Split Conformal Prediction for Medical Diagnostics:
Provides finite-sample coverage guarantees. Outputs a prediction set rather than
a point prediction:
P(Y in C(X)) >= 1 - alpha (e.g., 95% confidence).
Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2021.
"""
import numpy as np


class ConformalPredictor:
    """
    Computes conformal non-conformity quantile on a held-out calibration set.
    At test time, returns the set of all classes whose conformity score exceeds the threshold.
    """

    def __init__(self, alpha=0.05, class_names=None):
        self.alpha = alpha
        self.class_names = class_names or ["Meningioma", "Glioma", "Pituitary"]
        self.q_hat = 0.95  # Default threshold before calibration

    def calibrate(self, calib_probs, calib_labels):
        """
        calib_probs: np.ndarray [N, num_classes]
        calib_labels: np.ndarray [N] with true class indices
        """
        n = len(calib_labels)
        # Non-conformity score: 1 - prob[true_class]
        true_class_probs = calib_probs[np.arange(n), calib_labels]
        scores = 1.0 - true_class_probs

        # Compute (1 - alpha) * (1 + 1/n) empirical quantile
        quantile_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_level = min(1.0, max(0.0, quantile_level))
        self.q_hat = float(np.quantile(scores, quantile_level, method="higher"))
        return self.q_hat

    def predict_set(self, probs):
        """
        probs: np.ndarray of shape [num_classes]
        Returns:
            prediction_set (list of str): Classes included in the 1 - alpha confidence band
            is_uncertain (bool): True if set has more than 1 class
        """
        included_indices = []
        for c in range(len(probs)):
            score = 1.0 - probs[c]
            if score <= self.q_hat:
                included_indices.append(c)

        # Adaptive fallback: if ambiguous or below strict cutoff, include classes until cumulative sum >= 1 - alpha
        # or if the top 2 classes have close probabilities (ambiguous borderline)
        sorted_indices = np.argsort(probs)[::-1]
        if len(included_indices) == 0:
            cum_sum = 0.0
            target_coverage = 1.0 - self.alpha
            for idx in sorted_indices:
                included_indices.append(int(idx))
                cum_sum += probs[idx]
                if cum_sum >= target_coverage:
                    break

        # If top 2 are within 0.10 margin of each other, ensure both are included for clinical safety
        if len(sorted_indices) >= 2:
            top1, top2 = sorted_indices[0], sorted_indices[1]
            if abs(probs[top1] - probs[top2]) < 0.10 and top2 not in included_indices:
                included_indices.append(int(top2))

        prediction_set = [self.class_names[i] for i in sorted(included_indices)]
        is_uncertain = len(prediction_set) > 1

        return {
            "prediction_set": prediction_set,
            "confidence_level": 1.0 - self.alpha,
            "is_uncertain": is_uncertain,
            "conformal_quantile": self.q_hat,
        }
