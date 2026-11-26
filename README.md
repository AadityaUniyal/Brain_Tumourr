<div align="center">

# 🧠 NeuroVision Elite: Dual-Task Brain Tumor AI
### Clinical-Grade Segmentation, Diagnostic Subtyping, Conformal Uncertainty & Explainability

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20REST-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![CI Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![CLAIM Compliant](https://img.shields.io/badge/CLAIM-Compliant%20Checklist-purple?style=for-the-badge)](docs/CLAIM_CHECKLIST.md)

<p align="center">
  <b>A production-grade medical AI system built on strict zero-patient-leakage validation, joint multi-task learning (U-Net Segmentation + Multi-class Classification), energy-based Out-of-Distribution safety gates, and finite-sample Conformal Prediction.</b>
</p>

[✨ Live Web Cockpit](#-radiologist-cockpit-ui) • [🔬 Scientific Rigor](#-the-zero-leakage-mandate) • [🏗️ Architecture](#-system-architecture) • [🚀 Quickstart](#-getting-started) • [📋 CLAIM Checklist](docs/CLAIM_CHECKLIST.md) • [📄 Model Card](docs/MODEL_CARD.md)

</div>

---

## ⚠️ Medical AI Disclaimer
> **FOR INVESTIGATIONAL & RESEARCH PURPOSES ONLY.** This software is an assistive clinical decision support tool designed to aid qualified radiologists. It does not replace independent diagnosis by a licensed medical practitioner.

---

## 🔬 The Zero-Leakage & Real-Verification Mandate

Most open-source brain tumor repositories suffer from a critical flaw: **random slice splitting**.
In datasets such as the Jun Cheng Figshare cohort (3,064 slices across 233 patients), random splitting mixes slices from the same patient into both training and test sets. The model simply memorizes the patient's individual skull and ventricular geometry, reporting an inflated **"99% accuracy"** that collapses when exposed to a new, unseen patient.

### NeuroVision Elite implements:
1. **Strict Patient-Level Partitioning (`StratifiedGroupKFold`)**: 100% of an individual patient's anatomical slices reside exclusively in either the train or validation/test fold.
2. **Multi-Task Dual Head**: The network is trained simultaneously on:
   - **Branch A (Segmentation)**: Delineating exact tumor margins with soft Dice Loss.
   - **Branch B (Diagnostic Classification)**: Multi-class subtyping (Meningioma, Glioma, Pituitary) with Focal + Cross-Entropy Loss.
3. **Out-of-Distribution (OOD) Safety Gate**: Evaluates Helmholtz free energy $E(x) = -T \log \sum e^{f_i(x)/T}$. Non-brain scans or corrupted inputs are rejected with an explicit clinical safety warning.
4. **Split Conformal Prediction**: Replaces fragile point estimates with mathematically guaranteed 95% coverage prediction sets.

---

## 🏗️ System Architecture

```
User Upload (DICOM .dcm / NIfTI .nii.gz / MAT .mat / JPEG / PNG)
                            │
                            ▼
           ┌───────────────────────────────────┐
           │   MedicalImagePreprocessor        │
           │ • Dynamic Min-Max Normalization   │
           │ • Axial Slice Extraction (NIfTI)  │
           │ • Rescale Slope/Intercept (DICOM) │
           └───────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │    DualHeadBrainTumorNet    │
              │   (Deep ResNet50 Backbone)  │
              └──────────────┬──────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Segmentation Decoder   │       │   Classification Head   │
│ • Skip-connection FPN   │       │ • Global Avg Pool       │
│ • 1x1 Conv Logits       │       │ • LayerNorm + Dropout   │
│ • Soft Dice Mask Loss   │       │ • Cross-Entropy + Focal │
└───────────┬─────────────┘       └───────────┬─────────────┘
            │                                 │
            ▼                                 ▼
   Binary Tumor Mask &               Logits & Free Energy
Tumor Volume ($cm^3$)                     OOD Check
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Trustworthy AI Post-Process │
              │ • Grad-CAM Visual Heatmap   │
              │ • Conformal 95% Set         │
              │ • Volume Metrology ($cm^3$) │
              └──────────────┬──────────────┘
                             │
                             ▼
         Interactive Radiologist Cockpit / JSON API
```

---

## 🖥️ Radiologist Cockpit UI

The built-in web cockpit runs locally or in containerized environments:

- **Side-by-Side Visualization**: Seamless toggling between **Grad-CAM Saliency Maps** and **Tumor Segmentation Boundaries**.
- **Volumetric Metrology**: Automatically estimates tumor area ($mm^2$) and slice volume ($cm^3$) based on configurable voxel spacing.
- **Safety Indicator**: Real-time Out-of-Distribution badge alerting clinicians if the scan distribution is invalid.
- **1-Click PDF Clinical Report**: Formats print-ready diagnostic summaries with institutional headers and disclaimers.

---

## 🚀 Getting Started

### 1. Clone & Setup Environment

```bash
git clone https://github.com/AadityaUniyal/Brain_Tumourr.git
cd Brain_Tumourr

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / macOS

# Install package and production dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Verification Tests

```bash
pytest tests/ -v
```
*Executes all 8 unit and integration tests verifying the zero-leakage split engine, dual-head network, compound loss backprop, OOD energy detector, and FastAPI endpoints.*

### 3. Launch the Production API & Web Cockpit

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Navigate to:
- **Radiologist Cockpit UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI / Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Docker Deployment

Run the complete clinical service in a self-contained container:

```bash
docker compose up --build -d
```

Check service telemetry:
```bash
curl http://localhost:8000/health
```

---

## 📁 Repository Structure

```
Brain_Tumourr/
├── api/
│   ├── routes/             # REST endpoints (/diagnose, /health)
│   ├── schemas/            # Pydantic request/response validation
│   └── app.py              # FastAPI application factory
├── configs/
│   └── train_config.yaml   # Hyperparameters & loss weights
├── data/
│   ├── raw/                # Unprocessed DICOM/NIfTI/MAT storage
│   ├── processed/          # Standardized tensor caches
│   └── test_samples/       # Verified sample MRI scans
├── docs/
│   ├── CLAIM_CHECKLIST.md  # 42-item Medical AI compliance checklist
│   └── MODEL_CARD.md       # Mitchell et al. Model Card
├── frontend/               # Radiologist Cockpit (HTML/CSS/JS)
├── src/
│   ├── data/               # Preprocessor, dataset loader, split engine
│   ├── models/             # DualHeadBrainTumorNet, Dice/Focal losses
│   ├── training/           # Multi-task trainer, clinical metrics
│   └── inference/          # Predictor, Grad-CAM, OOD detector, Conformal
├── tests/                  # PyTest automated test suite
├── Dockerfile              # Production container build
├── docker-compose.yml      # Container orchestration
└── setup.py                # Installable Python package
```

---

## 📚 References & Standards
1. **Cheng et al.**, *"Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition"*, PLoS ONE, 2015.
2. **Mongan et al.**, *"Checklist for Artificial Intelligence in Medical Imaging (CLAIM)"*, Radiology: AI, 2020.
3. **Liu et al.**, *"Energy-based Out-of-distribution Detection"*, NeurIPS, 2020.
4. **Angelopoulos & Bates**, *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"*, 2021.
5. **Selvaraju et al.**, *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization"*, ICCV, 2017.

---

<div align="center">
Developed with scientific rigor by <b>Aaditya Uniyal</b>.
</div>
