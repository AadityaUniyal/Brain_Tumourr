"""
Phase 7 Unit & Integration Tests:
1. ComBat Scanner Harmonization (NeuroCombatHarmonizer)
2. Native DICOM Header Parsing & Volumetric Calculation (DICOMParser)
3. Automated Tumor Board PDF Dossier Generation (TumorBoardReportGenerator)
4. FastAPI Endpoint Integration (/api/v1/export-report)
"""
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data.harmonization import NeuroCombatHarmonizer
from src.data.dicom_parser import DICOMParser
from src.reporting.tumor_board_report import TumorBoardReportGenerator
from api.app import app

client = TestClient(app)

def test_neurocombat_harmonizer_variance_reduction():
    # Simulate 100 patient scans across 2 scanner sites (1.5T GE vs 3.0T Siemens)
    np.random.seed(42)
    n_samples = 100
    n_features = 12

    # Biological signal (Age effect)
    age = np.random.uniform(30, 75, size=(n_samples, 1))
    bio_effect = age @ np.random.normal(0.5, 0.1, size=(1, n_features))

    # Scanner batch labels (0 = 1.5T, 1 = 3.0T)
    batch = np.array([0] * 50 + [1] * 50)
    # Add scanner shift to batch 1
    scanner_shift = np.zeros((n_samples, n_features))
    scanner_shift[50:, :] += 2.5

    # Composite feature matrix
    noise = np.random.normal(0, 0.5, size=(n_samples, n_features))
    X_raw = bio_effect + scanner_shift + noise

    harmonizer = NeuroCombatHarmonizer(eb_iterations=15)
    X_harm = harmonizer.fit_transform(X_raw, batch, covariates=age)

    assert X_harm.shape == X_raw.shape
    assert not np.isnan(X_harm).any()

    # Verify ANOVA F-test reduction
    eval_res = harmonizer.evaluate_batch_disparity(X_raw, X_harm, batch)
    assert eval_res["mean_f_harm"] < eval_res["mean_f_raw"]
    assert eval_res["variance_reduction_pct"] > 60.0

def test_dicom_parser_volume_calculation():
    # 1000 voxels with 1.0mm x 1.0mm spacing and 3.0mm thickness
    # Volume = 1000 * 1.0 * 1.0 * 3.0 / 1000 = 3.0 cm³
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 0:100] = 1  # 10 x 100 = 1000 pixels

    vol = DICOMParser.calculate_physical_volume_cm3(
        mask, pixel_spacing_mm=(1.0, 1.0), slice_thickness_mm=3.0
    )
    assert pytest.approx(vol, 0.01) == 3.00

def test_tumor_board_pdf_generation():
    generator = TumorBoardReportGenerator()
    raw_mri = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
    seg_mask = np.zeros((128, 128), dtype=np.uint8)
    seg_mask[40:80, 40:80] = 1
    gradcam = np.random.uniform(0, 1, size=(128, 128)).astype(np.float32)

    case_data = {
        "patient_id": "PT-TEST-001",
        "patient_age": "48",
        "patient_sex": "FEMALE",
        "modality": "MR T1-CE Axial",
        "manufacturer": "SIEMENS (3.0T Magnetom Vida)",
        "slice_thickness_mm": 2.5,
        "tumor_volume_cm3": 8.42,
        "predicted_class": "Glioma",
        "confidence": 0.945,
        "who_2021_subtype": "Astrocytoma, IDH-mutant (CNS WHO Grade 3)",
        "idh_mutation_probability": 0.912,
        "idh_status": "IDH-Mutant",
        "codeletion_1p19q_probability": 0.082,
        "codeletion_1p19q_status": "1p/19q Intact (Non-codeleted)",
        "deepsurv_hazard_score": -0.42,
        "risk_tier": "Low Risk",
        "median_overall_survival_months": 54.0,
        "survival_curve": [{"month": m, "survival_prob": max(0.0, 1.0 - m*0.012)} for m in range(0, 61, 12)],
        "conformal_set": ["Glioma"],
        "energy_score": -8.45,
        "selective_signoff": True,
        "triage_priority": "NONE"
    }

    pdf_bytes = generator.generate_dossier_pdf(case_data, raw_mri, seg_mask, gradcam)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF-")

def test_export_report_api_endpoint():
    # Create synthetic MRI image in memory
    from PIL import Image
    synthetic_img = Image.new("L", (128, 128), color=128)
    img_bytes = io.BytesIO()
    synthetic_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("test_mri.png", img_bytes.getvalue(), "image/png")}
    response = client.post(
        "/api/v1/export-report?patient_age=50&patient_sex=MALE",
        files=files
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")
    assert len(response.content) > 5000
