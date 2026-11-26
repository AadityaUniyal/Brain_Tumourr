# Model Card: NeuroVision Elite Dual-Task Brain Tumor AI

Following the standardized framework proposed by Mitchell et al. (2019) in *"Model Cards for Model Reporting"*.

---

## 1. Model Details
- **Model Name**: NeuroVision Elite v2.0
- **Developers**: Aaditya Uniyal
- **Model Date**: November 2026
- **Architecture**: Dual-Head Multi-Task Convolutional Network (ResNet50 backbone + Feature Pyramid Semantic Decoder + MLP Classifier Head)
- **License**: MIT
- **Contact / Repository**: [https://github.com/AadityaUniyal/Brain_Tumourr](https://github.com/AadityaUniyal/Brain_Tumourr)

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
- **Zero Patient Data Leakage Guarantee**: 100% of an individual patient's anatomical slices reside exclusively in either the train or validation/test fold.
- **Image Preprocessing**:
  - Intensity Standardization: Dynamic Min-Max scaling to $[0, 1]$.
  - Spatial Normalization: Isotropic resampling and bilinear interpolation to $256 \times 256$.
  - Medical Augmentation: Horizontal flipping (exploiting brain anatomical symmetry), minor rotations ($\pm 15^\circ$), and scanner contrast jitter.

---

## 4. Evaluation Metrics & Scientific Transparency
The model reports genuine, reproducible metrics computed on held-out patients:
- **Macro F1-Score**: Harmonic mean of precision and recall across all 3 classes without prevalence skew.
- **Balanced Accuracy**: Macro-average recall across Meningioma, Glioma, and Pituitary classes.
- **Mean Dice Similarity Coefficient (DSC)**: Overlap between segmented lesion mask and radiologist ground truth.
- **Expected Calibration Error (ECE)**: Difference between predicted confidence and empirical accuracy across 10 probability bins.
- **Energy-Based Out-of-Distribution (OOD) Score**: Helmholtz free energy test rejecting non-brain scans ($\tau_{OOD} = -6.5$).

---

## 5. Ethical & Safety Considerations
- **Human-in-the-Loop Safeguard**: The system explicitly flags borderline diagnostic cases where the 95% conformal prediction set contains multiple classes (e.g. `{Glioma, Meningioma}`).
- **HIPAA Compliance**: The API operates entirely in-memory with zero unencrypted persistent disk writes for uploaded patient DICOM scans.
- **Bias & Fairness**: Scanner manufacturer variation (GE, Siemens, Philips) can introduce field inhomogeneities; users are advised to verify that magnetic field strength (1.5T vs 3.0T) aligns with dataset distribution.
