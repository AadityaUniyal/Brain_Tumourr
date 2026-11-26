# Model Card: NeuroVision Elite Dual-Task Brain Tumor AI

Following the standardized framework proposed by Mitchell et al. (2019) in *"Model Cards for Model Reporting"*.

---

## 1. Model Details
- **Model Name**: NeuroVision Elite v2.0
- **Developer**: Aaditya Uniyal
- **Model Date**: November 2026
- **Architecture**: Dual-Head Multi-Task Convolutional Network (ResNet50 backbone + Feature Pyramid Semantic Decoder + MLP Classifier Head)
- **License**: MIT
- **Repository**: [https://github.com/AadityaUniyal/Brain_Tumourr](https://github.com/AadityaUniyal/Brain_Tumourr)
- **Live Frontend**: [https://neurovision-elite-cockpit.vercel.app](https://neurovision-elite-cockpit.vercel.app)

---

## 2. Intended Clinical Use
- **Primary Use**: Assistive decision support for radiologists and neuro-oncologists during brain MRI evaluation.
- **Tasks**:
  1. Semantic segmentation of brain lesions (binary tumor boundary delineation).
  2. Multi-class tumor classification (Glioma, Meningioma, Pituitary Adenoma).
  3. In-plane tumor area ($mm^2$) and slice volumetric quantification ($cm^3$).
  4. Saliency visual attribution (Grad-CAM).
  5. Statistical uncertainty quantification via Conformal Prediction ($1 - \alpha = 0.95$ coverage).
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

---

## 5. Ethical & Safety Considerations
- **Human-in-the-Loop Safeguard**: The system explicitly flags borderline diagnostic cases where the 95% conformal prediction set contains multiple classes (e.g. `{Glioma, Meningioma}`).
- **Out-of-Distribution (OOD) Safety**: Energy-based detector rejects corrupted or non-brain scans ($\tau_{OOD} = -6.5$).
- **HIPAA Compliance**: The API operates entirely in-memory with zero unencrypted persistent disk writes for uploaded patient DICOM scans.
- **Bias & Fairness**: Scanner manufacturer variation (GE, Siemens, Philips) can introduce field inhomogeneities; users are advised to verify that magnetic field strength (1.5T vs 3.0T) aligns with dataset distribution.
