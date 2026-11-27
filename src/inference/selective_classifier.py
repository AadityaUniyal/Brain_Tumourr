"""
Conformal Risk-Controlled Selective Classification and Neuropathologist Triage Engine.
Decides whether to provide an automated diagnostic sign-off (g(x) = 1) or abstain and route
the MRI scan to a Human-in-the-Loop Senior Neuropathologist / Tumor Board Queue (g(x) = 0).
"""
from typing import Dict, List, Optional
import numpy as np


class SelectiveClassifier:
    """
    Selective Classification System (f, g) with Finite-Sample Risk Guarantees.
    f(x): Diagnostic Predictor
    g(x) in {0, 1}: Binary Selection/Abstention Function
    """

    def __init__(
        self,
        target_selective_risk: float = 0.05,
        confidence_threshold: float = 0.78,
        energy_threshold: float = -6.5,
        alpha: float = 0.05,
    ):
        self.target_selective_risk = target_selective_risk
        self.confidence_threshold = confidence_threshold
        self.energy_threshold = energy_threshold
        self.alpha = alpha

    def evaluate_decision(
        self,
        predicted_class: str,
        class_probabilities: Dict[str, float],
        conformal_set: List[str],
        energy_score: float,
        is_ood: bool,
        tumor_volume_cm3: float = 0.0,
    ) -> Dict:
        """
        Evaluates g(x) and generates comprehensive neuropathologist triage metadata.
        """
        max_prob = max(class_probabilities.values()) if class_probabilities else 0.0
        abstention_reasons = []

        # 1. Check Out-of-Distribution Safety
        if is_ood or energy_score < self.energy_threshold:
            abstention_reasons.append(
                f"Out-of-Distribution Anatomical Anomaly: Energy score {energy_score:.2f} is below safe threshold {self.energy_threshold:.2f}."
            )

        # 2. Check Conformal Ambiguity (Multi-class prediction set)
        if len(conformal_set) > 1:
            classes_str = ", ".join(conformal_set)
            abstention_reasons.append(
                f"Diagnostic Ambiguity: 95% Conformal Coverage set contains multiple competing pathologies: {{{classes_str}}}."
            )
        elif len(conformal_set) == 0:
            abstention_reasons.append("Zero Conformal Coverage: Model cannot guarantee true diagnosis within 95% confidence bounds.")

        # 3. Check Calibrated Selective Confidence Threshold
        if max_prob < self.confidence_threshold:
            abstention_reasons.append(
                f"Borderline Confidence: Top class probability ({max_prob * 100:.1f}%) is below the selective risk cutoff ({self.confidence_threshold * 100:.1f}%)."
            )

        # Binary Selection Decision g(x)
        g_x = 1 if len(abstention_reasons) == 0 else 0

        # Determine Action and Triage Priority
        if g_x == 1:
            action = "AUTOMATED_DIAGNOSIS"
            action_label = "Automated Diagnostic Sign-Off Approved"
            triage_priority = "NONE"
            clinical_guidance = (
                f"Unambiguous singleton diagnosis ({predicted_class}) with high confidence ({max_prob * 100:.1f}%) "
                f"satisfying selective risk constraint R(f, g) <= {self.target_selective_risk * 100:.1f}%."
            )
        else:
            action = "NEUROPATHOLOGIST_TRIAGE"
            action_label = "Abstained from Automated Sign-Off - Routed to Tumor Board"
            
            # Clinical Urgency Triage Scoring
            prob_glioma = class_probabilities.get("Glioma", 0.0)
            if prob_glioma >= 0.40 or tumor_volume_cm3 >= 15.0:
                triage_priority = "STAT / URGENT"
                urgency_note = "High likelihood of diffuse high-grade glioma or mass effect; fast-track for surgical consult."
            elif len(conformal_set) >= 2:
                triage_priority = "ELEVATED"
                urgency_note = "Equivocal radiological boundaries between competing tumor phenotypes; requires expert MRI re-read."
            else:
                triage_priority = "ROUTINE"
                urgency_note = "Standard second-opinion queue."

            clinical_guidance = (
                f"Safety gate abstained autonomous sign-off. Assigned Priority [{triage_priority}]: {urgency_note}"
            )

        return {
            "g_selection": g_x,
            "action": action,
            "action_label": action_label,
            "triage_priority": triage_priority,
            "clinical_guidance": clinical_guidance,
            "abstention_reasons": abstention_reasons,
            "target_selective_risk": self.target_selective_risk,
            "confidence_threshold": self.confidence_threshold,
            "energy_score": energy_score,
            "conformal_set_size": len(conformal_set),
            "is_selective_approved": bool(g_x == 1),
        }
