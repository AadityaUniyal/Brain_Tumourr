# 🏗️ Sulcus Clinical AI: System Architecture & Infrastructure Specification

**Document Version:** 2.4.0  
**Target Environment:** Amazon Web Services (AWS) ap-south-1 (Mumbai)  
**Security Standard:** HIPAA Security Rule & FDA SaMD (Software as a Medical Device) Track  
**Live Production URL:** [https://d2j24vbx45dlcs.cloudfront.net](https://d2j24vbx45dlcs.cloudfront.net)

---

## 1. Executive Architecture Overview

Sulcus is a zero-leakage, dual-task neuro-oncology operating system and radiogenomics engine. It bridges preoperative axial T1-CE MRI studies, quantitative IBSI radiomics, WHO CNS5 2021 molecular biomarkers, and dynamic time-to-event survival horizons into a unified, secure cloud environment.

```
                                  [ CLINICAL CLIENT ]
               (Neurosurgeons / Neuroradiologists / Tumor Boards)
                                         │
                                         ▼ HTTPS (TLS 1.3)
                       ┌───────────────────────────────────┐
                       │   Amazon CloudFront CDN (Edge)    │
                       │   d2j24vbx45dlcs.cloudfront.net   │
                       └─────────────────┬─────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
     Path: /*      ▼ (Static Assets via OAC)                   ▼ Path: /api/* (Dynamic)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  Amazon S3 Frontend Origin Bucket    │     │      Amazon EC2 Production Backend   │
│  sulcus-frontend-origin-145023097151 │     │  Instance: i-02e6d2ffb57a68f58 (t3)  │
│  - Polpharma Biologics Web App       │     │  - Nginx Reverse Proxy (Port 80)     │
│  - 3D Biologics Bottle Motives       │     │  - FastAPI / Uvicorn (Port 8000)     │
│  - Lenis Smooth Scroll Engine        │     │  - PyTorch DualHead / DeepSurv Engine│
└──────────────────────────────────────┘     └──────────────────┬───────────────────┘
                                                                │
                                        ┌───────────────────────┴───────────────────────┐
                                        ▼                                               ▼
                     ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
                     │     Amazon DynamoDB NoSQL Vault      │        │   Amazon S3 Private Imaging Vault    │
                     │  - sulcus_patient_records (Studies)  │        │  sulcus-imaging-vault-145023097151   │
                     │  - sulcus_audit_events (Audit Trail) │        │  - Raw DICOM / NIfTI Files (AES-256) │
                     └──────────────────────────────────────┘        └──────────────────────────────────────┘
```

---

## 2. Cloud Infrastructure & Hosting Topology

### 2.1 Amazon CloudFront (Global CDN & Edge Routing)
- **Distribution ID:** `E1W0SX18XDDAGL`
- **Domain Name:** `d2j24vbx45dlcs.cloudfront.net`
- **Behaviors:**
  - `Default (*)`: Routes to private S3 frontend bucket via **Origin Access Control (OAC)** with HTTP to HTTPS auto-redirect and Gzip/Brotli compression.
  - `/api/*`: Directly routes to the Amazon EC2 backend instance (`3.111.170.6`), passing all query strings, headers, and multipart form payloads with zero caching (`Cache-Control: no-cache, no-store`).

### 2.2 Amazon S3 Storage Architecture
1. **Frontend Origin Bucket (`sulcus-frontend-origin-145023097151`)**:
   - Host to the Polpharma Biologics design system, HTML5 interactive canvases, WebP/PNG 3D bottles, and Lenis inertial scroll runtime.
   - Access controlled strictly via CloudFront OAC policy (`SigV4`); direct public S3 website endpoints are disabled to prevent bucket enumeration.
2. **Clinical Imaging Vault Bucket (`sulcus-imaging-vault-145023097151`)**:
   - Zero-trust clinical DICOM store with server-side encryption (`AES256`).
   - Immutable audit tagging (`AccessAudit: True`, `Retention: 7-Years`).

### 2.3 Amazon EC2 Backend Compute
- **Instance ID:** `i-02e6d2ffb57a68f58`
- **Region & AZ:** `ap-south-1a`
- **Software Stack:**
  - Ubuntu 24.04 LTS (HVM)
  - Python 3.12 / PyTorch 2.6
  - Systemd Service: `sulcus.service`
  - High-performance Nginx reverse proxy buffering client multipart uploads.
- **IAM Instance Profile:** `sulcus-backend-instance-profile` with scoped permissions:
  - `dynamodb:PutItem`, `dynamodb:GetItem`, `dynamodb:Query`, `dynamodb:Scan`
  - `s3:PutObject`, `s3:GetObject` on `sulcus-imaging-vault-145023097151`
  - CloudWatch logs integration.

### 2.4 Amazon DynamoDB Clinical Registry
- **`sulcus_patient_records`**:
  - Partition Key: `patient_id` (String, e.g. `PT-17814`)
  - Sort Key: `timestamp` (String, ISO-8601)
  - Attributes: `predicted_subtype`, `confidence`, `rano_major_axis_mm`, `rano_minor_axis_mm`, `volume_cm3`, `idh_status`, `s3_vault_uri`.
- **`sulcus_audit_events`**:
  - Partition Key: `event_id` (String, UUIDv4)
  - Sort Key: `timestamp` (String)
  - Attributes: `user_npi`, `action`, `model_sha256`, `conformal_decision`, `ip_hash`.

---

## 3. Deep Learning & Radiogenomics Architecture

### 3.1 Dual-Head Morphological Network (`DualHeadNet`)
- **Backbone:** ResNet50 with custom 4-channel axial T1-CE receptive fields.
- **Head 1 (Segmentation):** Multi-scale attention-gated feature pyramid decoder outputting sub-pixel tumor masks ($256 \times 256$) with compound BCE-Dice loss:
  $$\mathcal{L}_{\text{seg}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot (1 - \text{Dice})$$
- **Head 2 (Diagnostic Classification):** Global average pooled spatial vectors routed through a 3-class differential classification head (Glioma, Meningioma, Pituitary Adenoma) trained with Temperature-calibrated Cross-Entropy.

### 3.2 Multimodal Radiogenomics & Survival Network (`MultimodalRadiogenomicsNet`)
- **Input Feats:** 48 Image Biomarker Standardization Initiative (IBSI) compliant radiomic features + 3 Tabular clinical covariates (Age, Sex, Surgical Extent).
- **Architecture:** Dual-stream cross-attention transformer layer followed by three task-specialized heads:
  1. **IDH Mutation Classifier:** Binary cross-entropy head predicting IDH1/2 wild-type vs. mutant status.
  2. **1p/19q Co-deletion Classifier:** Cost-sensitive classifier tuned with threshold $\tau = 0.47$ for rare-event detection.
  3. **DeepSurv Proportional Hazard Head:** Linear log-hazard estimation $g_\theta(x)$ trained via Cox partial likelihood with GroupDRO age-stratified loss:
     $$\mathcal{L}_{\text{Cox}} = - \sum_{i: E_i = 1} \left( g_\theta(x_i) - \log \sum_{j \in \mathcal{R}(T_i)} \exp(g_\theta(x_j)) \right)$$

---

## 4. UI/UX Polpharma Biologics Design System

The frontend experience is modeled after **Polpharma Biologics** (`https://www.polpharmabiologics.com/`):
- **Visual Vibe & Color Tokens:**
  - Deep Polpharma Teal: `#005453`
  - Vibrant Bio-Cyan: `#16aba9`
  - Soft Mint Accent: `#bce4e3`
  - Background Tint: `#e9f2f2`
  - Clean Charcoal: `#1c2222`
- **3D Bottle Motive & Transition Event:**
  - Floating 3D biologics vial rendered with real-time inertial scroll parallax (`translate3d` & rotation).
  - Fanned-out 3D bottle array in the evidence section (`.feature-shape_bottles`).
  - Fullscreen morphing curtain on authentication (`#authTransitionCurtain`) triggering staggered upward cascade entrances (`.cascade-in.delay-*`) on all workstation cockpit widgets.
- **Aspect-Ratio Pinned 4K Canvas:**
  - Prevents horizontal distortion or squishing during interactive split-wipe comparisons.
