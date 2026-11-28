# 📡 Sulcus Clinical AI: Production API Reference

**Protocol:** HTTPS / TLS 1.3  
**Base URL (CloudFront Global Edge):** `https://d2j24vbx45dlcs.cloudfront.net`  
**Direct Backend URL:** `http://3.111.170.6`  
**Content-Type:** `application/json` / `multipart/form-data`

---

## 1. System Health & Telemetry

### `GET /api/v1/health`
Returns infrastructure and model execution telemetry.

**Request:**
```bash
curl -X GET "https://d2j24vbx45dlcs.cloudfront.net/api/v1/health"
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "Sulcus Clinical AI",
  "version": "2.0.0",
  "device": "cpu",
  "gpu_available": false
}
```

---

## 2. Multi-Task Neural Diagnosis & RANO Metrology

### `POST /api/v1/diagnose`
Performs simultaneous sub-pixel tumor segmentation, 3-class differential diagnosis, conformal uncertainty quantification, and RANO 2D orthogonal calliper calculation.

**Request Parameters:**
- `file` (multipart/form-data, required): Raw axial T1-CE MRI scan (`.png`, `.jpg`, `.dcm`, `.mat`).

**Example Request:**
```bash
curl -X POST "https://d2j24vbx45dlcs.cloudfront.net/api/v1/diagnose" \
  -F "file=@samples/glioma_sample.png"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "diagnostic": {
    "predicted_tumor_type": "Glioma",
    "confidence": 0.942,
    "class_probabilities": {
      "Glioma": 0.942,
      "Meningioma": 0.041,
      "Pituitary": 0.017
    },
    "helmholtz_energy": -12.84,
    "conformal_prediction_set": ["Glioma"],
    "conformal_singleton": true,
    "triaging_status": "APPROVED_AUTOMATED_SIGN_OFF"
  },
  "segmentation": {
    "mask_rle": "...",
    "tumor_area_pixels": 1420,
    "estimated_volume_cm3": 34.82
  },
  "rano_metrology": {
    "major_diameter_mm": 41.2,
    "minor_diameter_mm": 28.6,
    "bidirectional_product_cm2": 11.78,
    "major_endpoints": [[148, 112], [253, 112]],
    "minor_endpoints": [[200, 75], [200, 149]]
  }
}
```

---

## 3. WHO CNS5 2021 Virtual Molecular Biopsy & Survival

### `POST /api/v1/radiogenomics`
Extracts 48 IBSI radiomic features, fuses clinical covariates, and estimates IDH mutation probability, 1p/19q co-deletion status, and DeepSurv non-linear hazard curves.

**Query Parameters:**
- `patient_age` (float, required): Age in years (20–90).
- `patient_sex` (string, required): `MALE` or `FEMALE`.
- `surgical_extent` (string, optional): `GTR` (Gross Total Resection) or `STR` (Subtotal Resection). Default: `GTR`.

**Example Request:**
```bash
curl -X POST "https://d2j24vbx45dlcs.cloudfront.net/api/v1/radiogenomics?patient_age=54&patient_sex=MALE" \
  -F "file=@samples/glioma_sample.png"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "who_2021_subtype": "Glioblastoma, IDH-wildtype (CNS WHO Grade 4)",
  "idh_mutation_probability": 0.082,
  "idh_status": "IDH-Wildtype (High Risk)",
  "codeletion_1p19q_probability": 0.011,
  "codeletion_1p19q_status": "1p/19q Intact",
  "deepsurv_hazard_score": 0.642,
  "risk_tier": "High Risk",
  "median_overall_survival_months": 14.8,
  "survival_curve": [
    {"month": 0.0, "survival_prob": 1.0},
    {"month": 12.0, "survival_prob": 0.58},
    {"month": 24.0, "survival_prob": 0.31},
    {"month": 36.0, "survival_prob": 0.16},
    {"month": 60.0, "survival_prob": 0.07}
  ]
}
```

---

## 4. Tumor Board Consultation Dossier Export

### `POST /api/v1/export-report`
Generates a signed, publication-grade PDF consultation dossier incorporating raw slice, neural segmentation mask, RANO callipers, survival prognosis, and SHA-256 model provenance.

**Response:** Binary PDF stream (`Content-Type: application/pdf`).

---

## 5. Longitudinal Patient Registry & Audit Trail

### `GET /api/v1/patient-history/{patient_id}`
Retrieves historical scans, longitudinal RANO volumetric deltas, and treatment response assessments stored in Amazon DynamoDB.

**Example Request:**
```bash
curl -X GET "https://d2j24vbx45dlcs.cloudfront.net/api/v1/patient-history/PT-8812"
```

**Response (200 OK):**
```json
{
  "patient_id": "PT-8812",
  "study_count": 3,
  "baseline_rano_cm2": 15.4,
  "current_rano_cm2": 11.78,
  "percent_change": -23.5,
  "response_category": "Partial Response (PR)",
  "vault_archive": "s3://sulcus-imaging-vault-145023097151/studies/PT-8812/"
}
```
