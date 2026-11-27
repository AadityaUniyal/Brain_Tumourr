# Model Card: NeuroVision Elite Dual-Task Brain Tumor AI

Following the standardized framework proposed by Mitchell et al. (2019) in *"Model Cards for Model Reporting"*.

---

## 1. Model Details
- **Model Name**: NeuroVision Elite v3.0 (Dual-Task Diagnostic & WHO 2021 Radiogenomics Survival Engine)
- **Developer**: Aaditya Uniyal
- **Model Date**: November 2026
- **Architecture**:
  1. **DualHeadBrainTumorNet**: ResNet50 encoder backbone + Feature Pyramid Skip-Decoder + Linear Diagnostic Head.
  2. **MultimodalRadiogenomicsNet**: Tri-Head Deep Survival Network fusing 48-dimensional IBSI high-throughput radiomics with clinical covariates (Age, Sex, Grade) via LayerNorm, GELU, and Gated Cross-Attention.
- **License**: MIT
- **Repository**: [https://github.com/AadityaUniyal/Brain_Tumourr](https://github.com/AadityaUniyal/Brain_Tumourr)
- **Live Frontend**: [https://neurovision-elite-cockpit.vercel.app](https://neurovision-elite-cockpit.vercel.app)

---

## 2. Intended Clinical Use
- **Primary Use**: Assistive decision support for neuro-oncologists, neurosurgeons, and radiologists during pre-operative brain MRI evaluation.
- **Tasks**:
  1. **Virtual Molecular Biopsy (WHO 2021 CNS5 Criteria)**: Non-invasive prediction of IDH1/2 mutation status and 1p/19q co-deletion.
  2. **Deep Survival Prognosis (DeepSurv)**: Right-censored time-to-event survival risk modeling using Cox Partial Likelihood loss.
  3. **Multi-task Morphological Segmentation**: Pixel-level tumor boundary delineation.
  4. **Multi-class Subtyping**: Differential diagnosis of Glioma vs Meningioma vs Pituitary Adenoma.
  5. **Conformal Uncertainty & OOD Detection**: Finite-sample 95% guaranteed coverage sets.
- **Out-of-Scope Use Cases**:
  - Autonomous clinical decision-making without board-certified radiologist review.
  - Non-brain radiological modalities (e.g., chest X-rays, abdominal CTs, mammograms).
  - Pediatric populations with rare embryonic neuro-neoplasms not present in adult training cohorts.

---

## 3. Data & Validation Strategy
- **Primary Source**: Jun Cheng Figshare Brain Tumor Dataset (3,064 T1-weighted Contrast-Enhanced MRI slices across 233 unique patients).
- **Validation Protocol**: 5-Fold `StratifiedGroupKFold` cross-validation partitioned strictly by `patient_id`.
- **Zero Patient Data Leakage Guarantee**: 100% of an individual patient's anatomical slices reside exclusively in either the train or validation/test fold (**0.0% leakage vs. 98.9% leakage in standard slice splits**).
- **Image Preprocessing**:
  - Intensity Standardization: Dynamic Min-Max scaling to $[0, 1]$.
  - Spatial Normalization: Bilinear interpolation to $128 \times 128$ for robust multi-task learning.
  - Medical Augmentation: Horizontal flipping (brain anatomical symmetry), rotations ($\pm 15^\circ$), and contrast jitter.

---

## 4. Quantitative Evaluation & Clinical Metrics

All metrics reported below were evaluated on a held-out test cohort of **474 MRI slices from 35 patients completely unseen during training**:

### Global Performance
- **Macro AUROC**: **0.9631** (Area under the Receiver Operating Characteristic curve)
- **Overall Accuracy**: **86.29%**
- **Balanced Accuracy**: **86.28%**
- **Macro F1-Score**: **85.59%**
- **Expected Calibration Error (ECE)**: **0.0187** (exceptionally calibrated confidence scores)
- **Mean Segmentation Dice (DSC)**: **0.4108**

### Per-Class Pathological Diagnostic Breakdown
- **Glioma**: Precision: **92.00%** | Sensitivity (Recall): **84.79%** | F1: **88.25%** ($N = 217$)
- **Pituitary Tumor**: Precision: **86.09%** | Sensitivity (Recall): **92.86%** | F1: **89.35%** ($N = 140$)
- **Meningioma**: Precision: **77.24%** | Sensitivity (Recall): **81.20%** | F1: **79.17%** ($N = 117$)

### Confusion Matrix on Held-Out Test Patients
```
                     PREDICTED CLASS
                Meningioma   Glioma   Pituitary
TRUE Meningioma     95         6         16
CLASS Glioma        28        184         5
     Pituitary       0         10        130
```

### Molecular Biomarker & Survival Prognostication (810 Validated TCGA Patients)
- **DeepSurv Harrell's C-Index**: **0.8151 ± 0.0133** (5-Fold Cross-Validation)
- **Kaplan-Meier Separation (Log-Rank Test)**: **$p = 1.54 \times 10^{-61}$** (High-Risk vs Low-Risk cohorts)
- **IDH1/2 Mutation AUROC**: **0.9256 ± 0.0230** (Macro-F1: 86.62%)
- **1p/19q Co-deletion AUROC**: **0.7265 ± 0.0358**
- **Loss Formulation**: Multi-task BCE + Cox Negative Partial Log-Likelihood ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{IDH}} + 1.2 \mathcal{L}_{1\text{p}19\text{q}} + 0.8 \mathcal{L}_{\text{Cox}}$)

---

## 5. Ethical & Safety Considerations
- **Human-in-the-Loop Safeguard**: The system explicitly flags borderline diagnostic cases where the 95% conformal prediction set contains multiple classes (e.g. `{Glioma, Meningioma}`).
- **Conformal Risk-Controlled Triaging**: Operating threshold $\tau^* = 0.400$ intercepts 89.71% of baseline diagnostic errors and routes them to a senior neuropathologist queue (`STAT / URGENT` or `ELEVATED`).
- **Out-of-Distribution (OOD) Safety**: Energy-based detector rejects corrupted or non-brain scans ($\tau_{OOD} = -6.5$).
- **Demographic Equity**: Gender parity gap strictly verified under fair clinical guidelines ($\Delta = 0.0011$).
- **HIPAA Compliance**: The API operates entirely in-memory with zero unencrypted persistent disk writes for uploaded patient DICOM scans.
- **Bias & Fairness**: Scanner manufacturer variation (GE, Siemens, Philips) can introduce field inhomogeneities; users are advised to verify that magnetic field strength (1.5T vs 3.0T) aligns with dataset distribution.

---

## 6. MLOps Governance & CI/CD Performance Regression Gating
- **Automated Regression Gate**: `scripts/ci_regression_gate.py` runs on every GitHub Pull Request via `.github/workflows/mlops_eval_gate.yml`.
- **8 Mandatory Clinical Pass Criteria**:
  1. Patient Test Data Leakage Rate $\le 0.00\%$ (`0.00%` PASS)
  2. Dual-Task Diagnostic Macro AUROC $\ge 0.9500$ (`0.9631` PASS)
  3. Dual-Task Diagnostic Accuracy $\ge 85.00\%$ (`86.29%` PASS)
  4. Expected Calibration Error (ECE) $\le 0.0500$ (`0.0187` PASS)
  5. DeepSurv Concordance Index $\ge 0.8000$ (`0.8151` PASS)
  6. Molecular Biopsy IDH AUROC $\ge 0.9000$ (`0.9256` PASS)
  7. Gender Disparity Gap $\Delta \le 0.0500$ (`0.0011` PASS)
  8. Conformal Selective Accuracy $\ge 95.00\%$ (`96.86%` PASS)
- **Model Registry (`models/model_registry.json`)**: Cryptographic tracking (SHA-256), versioning, and provenance logging for active production weights.
- **Reproducibility**: DVC pipeline (`dvc.yaml`) codifies complete data partitioning, radiomic feature extraction, multimodal survival training, and selective risk-coverage calibration.

---

## 7. Hospital Clinical Integration & Multi-Center Harmonization
- **Multi-Center Scanner Harmonization (`NeuroCombat`)**: Uses parametric Empirical Bayes to eliminate non-biological batch variance between 1.5T and 3.0T MRI protocols ($>70\%$ variance reduction).
- **Physical Volumetric Extraction**: Ingests native DICOM header tags (`PixelSpacing`, `SliceThickness`) to compute true anatomical tumor volume in $\text{cm}^3$.
- **Automated Tumor Board Dossier**: Generates clinical consultation PDFs (`/api/v1/export-report`) with tri-modal imaging plates, WHO CNS5 2021 molecular predictions, and DeepSurv survival trajectories.
