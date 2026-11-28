# CLAIM Checklist: Sulcus Clinical AI Compliance
*(Checklist for Artificial Intelligence in Medical Imaging — Mongan et al., Radiology: Artificial Intelligence 2020)*

| Section | Item # | Topic | Compliance Status in Sulcus Clinical AI |
| :--- | :--- | :--- | :--- |
| **Title / Abstract** | 1 | Identification as medical AI study | ✅ Clearly designated as dual-task medical AI for MRI segmentation, subtyping, and WHO 2021 molecular prognosis. |
| | 2 | Structured summary of objectives & results | ✅ Documented in README, Model Card, and experiment evaluation manifests. |
| **Introduction** | 3 | Clinical problem definition & target population | ✅ Adult intracranial neoplasms (Glioma, Meningioma, Pituitary) and molecular biomarkers (IDH1/2, 1p/19q). |
| | 4 | Intended clinical role | ✅ Second-reader / decision-support system for attending neuro-oncologists and radiologists. |
| **Methods: Data** | 5 | Data sources & recruitment | ✅ Jun Cheng Figshare cohort (233 patients, 3,064 T1-CE MRI slices) + TCGA GBMLGG (810 molecular/survival patients). |
| | 6 | Eligibility criteria | ✅ Axial T1-weighted contrast-enhanced brain MRI scans and biopsy-proven clinical molecular registries. |
| | 7 | Data de-identification & HIPAA | ✅ In-memory parsing; PHI DICOM tags stripped prior to inference. |
| | 8 | Ground truth reference standard | ✅ Radiologist-drawn discrete border coordinates & binary masks in `.mat`; genomic sequencing for IDH/1p19q. |
| | 9 | Data partitioning & leakage prevention | ✅ **Strict StratifiedGroupKFold by patient ID**. Zero patient slice leakage. |
| **Methods: Model** | 10 | Detailed model architecture | ✅ DualHeadBrainTumorNet (ResNet-34 + Attention-Gated U-Net Decoder + Classifier) + MultimodalRadiogenomicsNet (Tri-Head DeepSurv). |
| | 11 | Inputs, outputs, loss functions | ✅ Inputs: $128 \times 128 \times 3$ MRI + 48 IBSI radiomics + clinical covariates. Loss: CrossEntropy + SoftDice + FocalTversky + GroupDRO Cox. |
| | 12 | Hardware & software environment | ✅ PyTorch 2.0+, Python 3.10-3.13, CUDA/CPU optimized, locked requirements.txt. |
| **Methods: Evaluation** | 13 | Primary and secondary metrics | ✅ Macro F1, Balanced Accuracy, AUROC, Mean Dice, ECE, Harrell's C-index, 1p/19q F1, OOD Energy. |
| | 14 | Uncertainty quantification | ✅ Split Conformal Prediction (finite-sample coverage guarantees at $\alpha=0.05$) & selective sign-off. |
| | 15 | Explainability methods | ✅ High-resolution Grad-CAM on penultimate feature maps and patient-specific SHAP waterfall attributions. |
| **Results** | 16 | Demographic & clinical characteristics | ✅ Documented per patient, sex, and age distribution with GroupDRO disparity audit. |
| | 17 | Detailed performance reporting | ✅ Full confusion matrix, per-class precision/recall/F1, and CML 10-factor gate logged to JSON. |
| **Discussion** | 18 | Interpretation in clinical context | ✅ Addressed in Model Card; limitations of 2D slices vs 3D multi-sequence MRI noted. |
| | 19 | Failure analysis & OOD detection | ✅ Energy-based detector rejects non-brain images to prevent misdiagnosis; selective triage intercepts 96.88% of errors. |
