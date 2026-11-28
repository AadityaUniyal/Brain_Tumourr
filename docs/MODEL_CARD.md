# Model Card: Sulcus Clinical AI Dual-Task Brain Tumor Workstation

Following the standardized framework proposed by Mitchell et al. (2019) in *"Model Cards for Model Reporting"*.

---

## 1. Model Details
- **Model Name**: Sulcus Clinical AI v3.0 (Dual-Task Diagnostic & WHO 2021 Radiogenomics Survival Engine)
- **Developer**: Aaditya Uniyal
- **Model Date**: November 2026
- **Architecture**:
  1. **DualHeadBrainTumorNet**: Pretrained ResNet-34 encoder + Compact Bilinear Attention-Gated U-Net Decoder (`128->64->48->32->24`) + Linear Diagnostic Head.
  2. **MultimodalRadiogenomicsNet**: Tri-Head Deep Survival Network fusing 48-dimensional IBSI high-throughput radiomics with clinical covariates (Age, Sex, Grade) via LayerNorm, GELU, and Stratified GroupDRO Cox Partial Likelihood.
- **License**: MIT
- **Repository**: [https://github.com/AadityaUniyal/Brain_Tumourr](https://github.com/AadityaUniyal/Brain_Tumourr)
- **Live Production URL**: [https://d2j24vbx45dlcs.cloudfront.net](https://d2j24vbx45dlcs.cloudfront.net)
- **Cloud Architecture**: AWS CloudFront (Edge) + S3 (OAC) + EC2 (PyTorch/FastAPI) + DynamoDB (Registry)

---

## 2. Intended Clinical Use
- **Primary Use**: Assistive decision support for neuro-oncologists, neurosurgeons, and radiologists during pre-operative brain MRI evaluation.
- **Tasks**:
  1. **Virtual Molecular Biopsy (WHO 2021 CNS5 Criteria)**: Non-invasive prediction of IDH1/2 mutation status and 1p/19q co-deletion.
  2. **Deep Survival Prognosis (DeepSurv)**: Right-censored time-to-event survival risk modeling using GroupDRO age-stratified Cox loss.
  3. **Multi-task Morphological Segmentation**: Deep neural attention-gated lesion boundary delineation.
  4. **Multi-class Subtyping**: Differential diagnosis of Glioma vs Meningioma vs Pituitary Adenoma.
  5. **Conformal Uncertainty & OOD Detection**: Finite-sample 95% guaranteed coverage sets with automated selective sign-off.
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
  - Spatial Normalization: Bilinear interpolation to $128 \times 128$ for high-speed CPU training and inference.
  - Medical Augmentation: Horizontal flipping (brain anatomical symmetry), rotations ($\pm 15^\circ$), and contrast jitter.

---

## 4. Quantitative Evaluation & Clinical Metrics

All metrics reported below were evaluated on a held-out test cohort of **474 MRI slices from 35 patients completely unseen during training**:

### Global Performance
- **Macro AUROC**: **0.9775** (Area under the Receiver Operating Characteristic curve)
- **Overall Accuracy**: **88.19%**
- **Balanced Accuracy**: **86.88%**
- **Macro F1-Score**: **87.29%**
- **Expected Calibration Error (ECE)**: **0.0390** (Reliable uncertainty estimation)
- **Mean Segmentation Dice (DSC)**: **0.5276**
- **Selective Conformal Accuracy**: **98.91%** (on accepted scans, operating threshold $\tau^* = 0.400$)

### Per-Class Pathological Diagnostic Breakdown
- **Glioma**: Precision: **88.05%** | Sensitivity (Recall): **91.71%** | F1: **89.84%** ($N = 217$)
- **Pituitary Tumor**: Precision: **91.55%** | Sensitivity (Recall): **92.86%** | F1: **92.20%** ($N = 140$)
- **Meningioma**: Precision: **83.96%** | Sensitivity (Recall): **76.07%** | F1: **79.82%** ($N = 117$)

### Molecular Biomarker & Survival Prognostication (810 Validated TCGA Patients)
- **DeepSurv Harrell's C-Index**: **0.8192 ± 0.0218** (5-Fold Cross-Validation)
- **IDH1/2 Mutation AUROC**: **0.9184 ± 0.0197** (Macro-F1: **87.66%**)
- **1p/19q Co-deletion AUROC**: **0.7496 ± 0.0351** (Calibrated Threshold $\tau = 0.47$, F1: **49.76%**)
- **Demographic Gender Equity**: Disparity gap $\Delta = \mathbf{0.0245}$ (Verified fair under clinical guidelines $< 0.05$)
- **Senior (>60) C-index**: **0.7720** (boosted by intra-stratum demographic ranking loss)

---

## 5. Ethical & Safety Considerations
- **Human-in-the-Loop Safeguard**: The system explicitly flags borderline diagnostic cases where the 95% conformal prediction set contains multiple classes (e.g. `{Glioma, Meningioma}`).
- **Conformal Risk-Controlled Triaging**: Operating threshold $\tau^* = 0.400$ intercepts 96.88% of baseline diagnostic errors and routes them to a senior neuropathologist queue (`STAT / URGENT` or `ELEVATED`).
- **Scanner Physics Robustness**: Tested under simulated B1 bias field (0.1–0.5: AUROC 0.9817–0.9828), Rician noise ($\sigma \in [0.03, 0.15]$), and motion artifacts.
- **HIPAA Compliance**: The API operates entirely in-memory with zero unencrypted persistent disk writes for uploaded patient DICOM scans.

---

## 6. MLOps Governance & CI/CD Performance Regression Gating
- **Automated Regression Gate**: `scripts/ci_regression_gate.py` runs on every GitHub Pull Request via `.github/workflows/mlops_eval_gate.yml`.
- **10 Mandatory Clinical Pass Criteria**:
  1. Patient Test Data Leakage Rate $\le 0.00\%$ (`0.00%` PASS)
  2. Dual-Task Diagnostic Macro AUROC $\ge 0.9500$ (`0.9775` PASS)
  3. Dual-Task Diagnostic Accuracy $\ge 85.00\%$ (`88.19%` PASS)
  4. Expected Calibration Error (ECE) $\le 0.0500$ (`0.0390` PASS)
  5. DeepSurv Concordance Index $\ge 0.8000$ (`0.8192` PASS)
  6. Molecular Biopsy IDH AUROC $\ge 0.9000$ (`0.9184` PASS)
  7. 1p/19q Co-deletion F1 Score $\ge 0.4500$ (`0.4976` PASS)
  8. Demographic Gender Disparity Gap $\Delta \le 0.0500$ (`0.0245` PASS)
  9. Conformal Selective Accuracy $\ge 95.00\%$ (`98.91%` PASS)
  10. Scanner B1 Field Inhomogeneity Test AUROC $\ge 0.9500$ (`0.9848` PASS)
- **Model Registry (`models/model_registry.json`)**: Cryptographic tracking (SHA-256), versioning, and provenance logging for active production weights.
- **Reproducibility**: DVC pipeline (`dvc.yaml`) codifies complete data partitioning, radiomic feature extraction, multimodal survival training, and selective risk-coverage calibration.

---

## 7. Hospital Clinical Integration & Multi-Center Harmonization
- **Multi-Center Scanner Harmonization (`NeuroCombat`)**: Uses parametric Empirical Bayes to eliminate non-biological batch variance between 1.5T and 3.0T MRI protocols ($>70\%$ variance reduction).
- **RANO (Response Assessment in Neuro-Oncology) Metrology**: Computes 2D bidirectional orthogonal diameters ($D_{\max} \times D_{\perp}$) directly from spatial contours with pixel spacing calibration.
- **Automated Tumor Board PDF Dossier**: Publication-grade clinical report signed with cryptographic SHA-256 model provenance.
- **Polpharma Biologics Clinical Workstation**: High-performance responsive web client hosted on AWS CloudFront with 3D biologics vial motives, Lenis smooth scrolling, and multi-tab PACS telemetry.
- **Physical Volumetric Extraction**: Ingests native DICOM header tags (`PixelSpacing`, `SliceThickness`) to compute true anatomical tumor volume in $\text{cm}^3$.
- **Automated Tumor Board Dossier**: Generates clinical consultation PDFs (`/api/v1/export-report`) with tri-modal imaging plates, WHO CNS5 2021 molecular predictions, and DeepSurv survival trajectories.
