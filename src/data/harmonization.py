"""
NeuroCombat Multi-Center Radiomic Harmonizer.
Implements Parametric Empirical Bayes (ComBat) for medical imaging / radiomic datasets.
Eliminates scanner-induced batch effects (1.5T vs. 3.0T field strengths, vendor drift)
while strictly preserving patient biological covariates (Age, Sex, Grade).

Reference:
- Fortin et al., Harmonization of multi-site diffusion tensor imaging data, NeuroImage (2017).
- Johnson et al., Adjusting batch effects in microarray expression data using empirical Bayes, Biostatistics (2007).
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union

class NeuroCombatHarmonizer:
    """
    Parametric Empirical Bayes (ComBat) batch-effect corrector for radiomics.
    """
    def __init__(self, eb_iterations: int = 30, eps: float = 1e-8):
        self.eb_iterations = eb_iterations
        self.eps = eps
        self.fitted_ = False
        self.grand_mean_ = None
        self.sigma_pooled_ = None
        self.beta_covariates_ = None
        self.covariates_mean_ = None
        self.gamma_star_ = None  # EB additive batch effect per batch x feature
        self.delta_star_ = None  # EB multiplicative batch effect per batch x feature
        self.batch_levels_ = []

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Union[np.ndarray, pd.Series, list],
        covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> "NeuroCombatHarmonizer":
        """
        Fits ComBat parameters on radiomic features X with batch labels and optional covariates.
        """
        X_mat = np.asarray(X, dtype=np.float64)
        batch_arr = np.asarray(batch)
        n_samples, n_features = X_mat.shape

        self.batch_levels_ = np.unique(batch_arr).tolist()
        n_batches = len(self.batch_levels_)
        batch_to_idx = {b: i for i, b in enumerate(self.batch_levels_)}
        sample_batch_idx = np.array([batch_to_idx[b] for b in batch_arr])

        # 1. Construct Batch Design Matrix B (n_samples, n_batches)
        B = np.zeros((n_samples, n_batches), dtype=np.float64)
        for i, b_idx in enumerate(sample_batch_idx):
            B[i, b_idx] = 1.0

        # 2. Add centered biological covariates
        if covariates is not None:
            cov_mat = np.asarray(covariates, dtype=np.float64)
            if cov_mat.ndim == 1:
                cov_mat = cov_mat.reshape(-1, 1)
            self.covariates_mean_ = np.mean(cov_mat, axis=0, keepdims=True)
            cov_centered = cov_mat - self.covariates_mean_
            M = np.column_stack([B, cov_centered])
        else:
            self.covariates_mean_ = None
            M = B

        # 3. OLS Estimation of batch intercepts and biological coefficients
        pinv_M = np.linalg.pinv(M)
        theta_hat = pinv_M @ X_mat  # shape: (n_batches + n_covs, n_features)

        batch_means = theta_hat[:n_batches, :]  # shape: (n_batches, n_features)
        if covariates is not None:
            self.beta_covariates_ = theta_hat[n_batches:, :]  # shape: (n_covs, n_features)
        else:
            self.beta_covariates_ = None

        # Grand mean across all batches weighted by batch size
        batch_counts = np.bincount(sample_batch_idx, minlength=n_batches)
        weights = batch_counts.reshape(-1, 1) / n_samples
        self.grand_mean_ = np.sum(batch_means * weights, axis=0)  # shape: (n_features,)

        # Residuals and pooled variance
        residuals = X_mat - (M @ theta_hat)
        df = max(1, n_samples - M.shape[1])
        var_pooled = np.sum(residuals ** 2, axis=0) / df
        var_pooled = np.maximum(var_pooled, self.eps)
        self.sigma_pooled_ = np.sqrt(var_pooled)

        # 4. Standardized data: remove grand mean and biological effects
        if covariates is not None and self.beta_covariates_ is not None:
            bio_signal = (cov_mat - self.covariates_mean_) @ self.beta_covariates_
        else:
            bio_signal = np.zeros((n_samples, n_features))

        # Standardized values Z where batch effect remains as location shift
        Z = (X_mat - self.grand_mean_ - bio_signal) / (self.sigma_pooled_ + self.eps)

        # 5. Method of Moments estimates for gamma and delta
        gamma_hat = np.zeros((n_batches, n_features))
        delta_hat = np.zeros((n_batches, n_features))

        for b_idx in range(n_batches):
            mask = (sample_batch_idx == b_idx)
            n_b = np.sum(mask)
            if n_b > 0:
                gamma_hat[b_idx, :] = np.mean(Z[mask, :], axis=0)
                var_b = np.var(Z[mask, :], axis=0, ddof=1) if n_b > 1 else np.ones(n_features)
                delta_hat[b_idx, :] = np.maximum(var_b, self.eps)

        # 6. Empirical Bayes Shrinkage
        self.gamma_star_ = np.zeros_like(gamma_hat)
        self.delta_star_ = np.zeros_like(delta_hat)

        for b_idx in range(n_batches):
            mask = (sample_batch_idx == b_idx)
            n_b = np.sum(mask)
            if n_b <= 1:
                self.gamma_star_[b_idx, :] = gamma_hat[b_idx, :]
                self.delta_star_[b_idx, :] = np.sqrt(delta_hat[b_idx, :])
                continue

            gamma_bar = np.mean(gamma_hat[b_idx, :])
            tau_sq = np.var(gamma_hat[b_idx, :], ddof=1) + self.eps

            delta_bar = np.mean(delta_hat[b_idx, :])
            delta_var = np.var(delta_hat[b_idx, :], ddof=1) + self.eps
            alpha_prior = (2.0 * delta_var + delta_bar ** 2) / (delta_var + self.eps)
            alpha_prior = max(alpha_prior, 2.0 + self.eps)
            beta_prior = delta_bar * (alpha_prior - 1.0)

            g_star = gamma_hat[b_idx, :].copy()
            d_star_sq = delta_hat[b_idx, :].copy()

            for _ in range(self.eb_iterations):
                g_star = (n_b * tau_sq * gamma_hat[b_idx, :] + d_star_sq * gamma_bar) / (
                    n_b * tau_sq + d_star_sq + self.eps
                )
                sum_sq = np.sum((Z[mask, :] - g_star) ** 2, axis=0)
                d_star_sq = (2.0 * beta_prior + sum_sq) / (2.0 * alpha_prior + n_b - 2.0 + self.eps)
                d_star_sq = np.maximum(d_star_sq, self.eps)

            self.gamma_star_[b_idx, :] = g_star
            self.delta_star_[b_idx, :] = np.sqrt(d_star_sq)

        self.fitted_ = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Union[np.ndarray, pd.Series, list],
        covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Harmonizes features using fitted Empirical Bayes batch parameters.
        """
        if not self.fitted_:
            raise RuntimeError("NeuroCombatHarmonizer must be fit before calling transform.")

        X_mat = np.asarray(X, dtype=np.float64)
        batch_arr = np.asarray(batch)
        n_samples, n_features = X_mat.shape

        batch_to_idx = {b: i for i, b in enumerate(self.batch_levels_)}

        if covariates is not None and self.beta_covariates_ is not None:
            cov_mat = np.asarray(covariates, dtype=np.float64)
            if cov_mat.ndim == 1:
                cov_mat = cov_mat.reshape(-1, 1)
            bio_signal = (cov_mat - self.covariates_mean_) @ self.beta_covariates_
        else:
            bio_signal = np.zeros((n_samples, n_features))

        X_harmonized = np.zeros_like(X_mat)

        for i in range(n_samples):
            b_label = batch_arr[i]
            if b_label in batch_to_idx:
                b_idx = batch_to_idx[b_label]
                g_star = self.gamma_star_[b_idx, :]
                d_star = self.delta_star_[b_idx, :]
            else:
                g_star = np.zeros(n_features)
                d_star = np.ones(n_features)

            # Standardized residual
            z_i = (X_mat[i, :] - self.grand_mean_ - bio_signal[i, :]) / (self.sigma_pooled_ + self.eps)
            # Remove batch effect and rescale
            z_harm = (z_i - g_star) / (d_star + self.eps)
            # Restore grand mean and biological signal
            X_harmonized[i, :] = z_harm * self.sigma_pooled_ + self.grand_mean_ + bio_signal[i, :]

        return X_harmonized

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        batch: Union[np.ndarray, pd.Series, list],
        covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        return self.fit(X, batch, covariates).transform(X, batch, covariates)

    def evaluate_batch_disparity(
        self,
        X_raw: np.ndarray,
        X_harm: np.ndarray,
        batch: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes ANOVA F-statistic reduction across scanner batches before and after ComBat.
        """
        from scipy import stats

        batch_arr = np.asarray(batch)
        batches = np.unique(batch_arr)
        if len(batches) < 2:
            return {"mean_f_raw": 0.0, "mean_f_harm": 0.0, "variance_reduction_pct": 100.0}

        f_raw_list = []
        f_harm_list = []

        for j in range(X_raw.shape[1]):
            groups_raw = [X_raw[batch_arr == b, j] for b in batches if np.sum(batch_arr == b) > 1]
            groups_harm = [X_harm[batch_arr == b, j] for b in batches if np.sum(batch_arr == b) > 1]

            if len(groups_raw) >= 2:
                f_r, _ = stats.f_oneway(*groups_raw)
                f_h, _ = stats.f_oneway(*groups_harm)
                if not np.isnan(f_r):
                    f_raw_list.append(f_r)
                if not np.isnan(f_h):
                    f_harm_list.append(f_h)

        mean_f_raw = float(np.mean(f_raw_list)) if f_raw_list else 1.0
        mean_f_harm = float(np.mean(f_harm_list)) if f_harm_list else 0.0
        reduction = max(0.0, (1.0 - mean_f_harm / (mean_f_raw + self.eps)) * 100.0)

        return {
            "mean_f_raw": round(mean_f_raw, 4),
            "mean_f_harm": round(mean_f_harm, 4),
            "variance_reduction_pct": round(reduction, 2)
        }
