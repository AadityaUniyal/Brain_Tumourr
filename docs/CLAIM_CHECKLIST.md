# CLAIM Checklist: NeuroVision Elite Compliance
*(Checklist for Artificial Intelligence in Medical Imaging — Mongan et al., Radiology: Artificial Intelligence 2020)*

| Section | Item # | Topic | Compliance Status in NeuroVision Elite |
| :--- | :--- | :--- | :--- |
| **Title / Abstract** | 1 | Identification as medical AI study | ✅ Clearly designated as dual-task medical AI for MRI segmentation & classification. |
| | 2 | Structured summary of objectives & results | ✅ Documented in README, Model Card, and experiment evaluation manifests. |
| **Introduction** | 3 | Clinical problem definition & target population | ✅ Adult intracranial neoplasms (Glioma, Meningioma, Pituitary). |
| | 4 | Intended clinical role | ✅ Second-reader / decision-support system for radiologists. |
| **Methods: Data** | 5 | Data sources & recruitment | ✅ Jun Cheng Figshare cohort (233 patients, 3,064 T1-CE MRI slices). |
| | 6 | Eligibility criteria | ✅ Axial T1-weighted contrast-enhanced brain MRI scans. |
| | 7 | Data de-identification & HIPAA | ✅ In-memory parsing; PHI DICOM tags stripped prior to inference. |
| | 8 | Ground truth reference standard | ✅ Radiologist-drawn discrete border coordinates & binary masks in `.mat`. |
| | 9 | Data partitioning & leakage prevention | ✅ **Strict StratifiedGroupKFold by patient ID**. Zero slice leakage. |
| **Methods: Model** | 10 | Detailed model architecture | ✅ DualHeadBrainTumorNet (ResNet50 + FPN Skip Decoder + Classifier). |
| | 11 | Inputs, outputs, loss functions | ✅ Inputs: $256 \times 256 \times 3$; Outputs: logits $[3]$ + mask $[1, 256, 256]$. Loss: CrossEntropy + Dice + Focal. |
| | 12 | Hardware & software environment | ✅ PyTorch 2.15, Python 3.13, CUDA optional, locked requirements.txt. |
| **Methods: Evaluation** | 13 | Primary and secondary metrics | ✅ Macro F1, Balanced Accuracy, AUROC, Mean Dice, ECE, OOD Energy. |
| | 14 | Uncertainty quantification | ✅ Split Conformal Prediction (finite-sample coverage guarantees at $\alpha=0.05$). |
| | 15 | Explainability methods | ✅ High-resolution Grad-CAM on penultimate feature maps. |
| **Results** | 16 | Demographic & clinical characteristics | ✅ Documented per patient and class distribution. |
| | 17 | Detailed performance reporting | ✅ Full confusion matrix and per-class precision/recall/F1 logged to JSON. |
| **Discussion** | 18 | Interpretation in clinical context | ✅ Addressed in Model Card; limitations of 2D slices vs 3D multi-sequence MRI noted. |
| | 19 | Failure analysis & OOD detection | ✅ Energy-based detector rejects non-brain images to prevent misdiagnosis. |
