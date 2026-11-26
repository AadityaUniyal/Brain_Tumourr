"""
Energy-Based Out-of-Distribution (OOD) Safety Detector:
Evaluates Helmholtz free energy on neural logits to detect and reject non-brain,
corrupted, or out-of-distribution inputs.
Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
"""
import torch
import numpy as np


class EnergyOODDetector:
    """
    Computes free energy:
    E(x; T) = -T * logsumexp(logits / T)
    In-distribution samples (brain MRIs) have low free energy (e.g. < -6.0).
    Out-of-distribution samples (non-brain, noise, random photos) have high free energy.
    """

    def __init__(self, energy_threshold=-6.5, temperature=1.0):
        self.energy_threshold = energy_threshold
        self.temperature = temperature

    def compute_energy(self, logits):
        """
        logits: Tensor of shape [B, num_classes] or [num_classes]
        Returns:
            energy: float or np.ndarray
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Free energy formula
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
        return energy.detach().cpu().numpy()

    def check(self, logits):
        """
        Returns:
            is_ood (bool): True if sample is Out-of-Distribution (Reject)
            energy_val (float)
            status_message (str)
        """
        energy_arr = self.compute_energy(logits)
        energy_val = float(energy_arr[0])

        is_ood = energy_val > self.energy_threshold
        if is_ood:
            msg = f"SAFETY ALERT: Input rejected as Out-of-Distribution (Free Energy: {energy_val:.2f} > {self.energy_threshold})."
        else:
            msg = f"Scan distribution verified (Free Energy: {energy_val:.2f} <= {self.energy_threshold})."

        return is_ood, energy_val, msg
