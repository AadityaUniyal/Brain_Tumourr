"""
Clinical Diagnostic and Radiogenomics Endpoints:
1. /diagnose: Multi-task segmentation, classification, OOD verification, and Grad-CAM generation.
2. /radiogenomics: WHO 2021 Virtual Molecular Biopsy (IDH, 1p/19q) & DeepSurv Dynamic Survival Prognosis.
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import Response

from api.schemas.models import DiagnosticResponse, RadiogenomicsResponse
from src.inference.predictor import ClinicalPredictor
from src.inference.radiogenomics_predictor import RadiogenomicsPredictor
from src.data.dicom_parser import DICOMParser
from src.reporting.tumor_board_report import TumorBoardReportGenerator

router = APIRouter(prefix="/api/v1", tags=["Diagnostic & Radiogenomics Inference"])

# Global singleton predictor instances
_predictor = None
_radiogenomics_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ClinicalPredictor()
    return _predictor


def get_radiogenomics_predictor():
    global _radiogenomics_predictor
    if _radiogenomics_predictor is None:
        _radiogenomics_predictor = RadiogenomicsPredictor()
    return _radiogenomics_predictor


def encode_image_base64(image_np, is_bgr=True):
    """Encodes numpy image array to base64 JPEG string."""
    if is_bgr:
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)

    pil_img = Image.fromarray(rgb_img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@router.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose_scan(
    file: UploadFile = File(..., description="MRI Scan in DICOM, NIfTI, MAT, or standard image format"),
    slice_thickness_mm: float = Query(3.0, description="MRI slice thickness in mm (for volume estimation)"),
    pixel_spacing_mm: float = Query(0.8, description="In-plane pixel spacing in mm"),
):
    try:
        content = await file.read()
        predictor = get_predictor()

        # Check for native DICOM tags
        dicom_meta = None
        if DICOMParser.is_dicom(content):
            try:
                _, d_meta = DICOMParser.parse_dicom(content)
                dicom_meta = d_meta
                pixel_spacing_mm = float(d_meta.get("pixel_spacing_mm", [0.8, 0.8])[0])
                slice_thickness_mm = float(d_meta.get("slice_thickness_mm", 3.0))
            except Exception:
                pass

        result = predictor.predict(
            file_source=content,
            filename=file.filename,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        # Base64 encode visualizations for frontend rendering
        cam_overlay_b64 = encode_image_base64(result["visualizations"]["gradcam_overlay"], is_bgr=True)
        mask_b64 = encode_image_base64((result["visualizations"]["seg_mask"] * 255).astype(np.uint8), is_bgr=False)

        return DiagnosticResponse(
            status="success",
            filename=file.filename,
            is_ood=result["is_ood"],
            ood_message=result["ood_message"],
            energy_score=result["energy_score"],
            diagnostic=result["diagnostic"],
            quantification=result["quantification"],
            selective_triage=result.get("selective_triage"),
            dicom_metadata=dicom_meta,
            gradcam_base64=cam_overlay_b64,
            mask_base64=mask_b64,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic pipeline error: {str(e)}")


@router.post("/radiogenomics", response_model=RadiogenomicsResponse)
async def analyze_radiogenomics(
    file: UploadFile = File(..., description="MRI Scan to extract IBSI radiomics"),
    patient_age: float = Query(52.0, description="Patient age at diagnosis in years"),
    patient_sex: str = Query("MALE", description="Biological sex (MALE or FEMALE)"),
    is_gbm: bool = Query(False, description="Whether scan exhibits Glioblastoma phenotype"),
):
    try:
        content = await file.read()
        rad_predictor = get_radiogenomics_predictor()

        # If DICOM, override default age/sex if available
        if DICOMParser.is_dicom(content):
            try:
                _, d_meta = DICOMParser.parse_dicom(content)
                if d_meta.get("patient_sex") in ["MALE", "FEMALE"]:
                    patient_sex = d_meta["patient_sex"]
                age_str = d_meta.get("patient_age", "")
                if age_str and age_str[:-1].isdigit():
                    patient_age = float(age_str[:-1])
            except Exception:
                pass

        result = rad_predictor.predict(
            file_source=content,
            filename=file.filename,
            age=patient_age,
            sex=patient_sex,
            is_gbm=is_gbm,
        )
        return RadiogenomicsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Radiogenomics pipeline error: {str(e)}")


@router.post("/export-report")
async def export_tumor_board_report(
    file: UploadFile = File(..., description="MRI Scan in DICOM, NIfTI, MAT, or standard image format"),
    patient_age: float = Query(52.0, description="Patient age at diagnosis in years"),
    patient_sex: str = Query("MALE", description="Biological sex (MALE or FEMALE)"),
    is_gbm: bool = Query(False, description="Whether scan exhibits Glioblastoma phenotype"),
    slice_thickness_mm: float = Query(3.0, description="Slice thickness in mm"),
    pixel_spacing_mm: float = Query(0.8, description="Pixel spacing in mm"),
):
    """
    Generates a publication-grade Multidisciplinary Neuro-Oncology Tumor Board Consultation PDF.
    Incorporates tri-modal visual plates, WHO 2021 molecular predictions, DeepSurv trajectory,
    conformal triage risk bounds, and cryptographic SHA-256 model provenance.
    """
    try:
        content = await file.read()
        predictor = get_predictor()
        rad_predictor = get_radiogenomics_predictor()

        # 1. Run diagnostic inference
        diag_res = predictor.predict(
            file_source=content,
            filename=file.filename,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        # 2. Run radiogenomics & DeepSurv inference
        is_gbm_inferred = is_gbm or (diag_res["diagnostic"]["predicted_tumor_type"] == "Glioma")
        rad_res = rad_predictor.predict(
            file_source=content,
            filename=file.filename,
            age=patient_age,
            sex=patient_sex,
            is_gbm=is_gbm_inferred,
        )

        # 3. Check DICOM metadata
        dicom_meta = {}
        if DICOMParser.is_dicom(content):
            try:
                _, dicom_meta = DICOMParser.parse_dicom(content)
            except Exception:
                pass

        # 4. Assemble case data
        case_data = {
            "patient_id": dicom_meta.get("patient_id", f"PT-{file.filename[:8].upper()}"),
            "patient_age": str(int(patient_age)),
            "patient_sex": patient_sex.upper(),
            "modality": dicom_meta.get("modality", "MR T1-CE Axial"),
            "manufacturer": dicom_meta.get("manufacturer", "SIEMENS Healthineers (3.0T Magnetom)"),
            "slice_thickness_mm": dicom_meta.get("slice_thickness_mm", slice_thickness_mm),
            "tumor_volume_cm3": diag_res["quantification"]["estimated_volume_cm3"],
            "predicted_class": diag_res["diagnostic"]["predicted_tumor_type"],
            "confidence": diag_res["diagnostic"]["confidence"],
            "who_2021_subtype": rad_res["who_2021_subtype"],
            "idh_mutation_probability": rad_res["idh_mutation_probability"],
            "idh_status": rad_res["idh_status"],
            "codeletion_1p19q_probability": rad_res["codeletion_1p19q_probability"],
            "codeletion_1p19q_status": rad_res["codeletion_1p19q_status"],
            "deepsurv_hazard_score": rad_res["deepsurv_hazard_score"],
            "risk_tier": rad_res["risk_tier"],
            "median_overall_survival_months": rad_res["median_overall_survival_months"],
            "survival_curve": [p.dict() if hasattr(p, "dict") else p for p in rad_res["survival_curve"]],
            "conformal_set": diag_res["diagnostic"]["conformal_prediction_set"],
            "energy_score": diag_res["energy_score"],
            "selective_signoff": diag_res.get("selective_triage", {}).get("is_selective_approved", True),
            "triage_priority": diag_res.get("selective_triage", {}).get("triage_priority", "STAT / URGENT")
        }

        # Visual plates
        raw_mri = diag_res["visualizations"].get("raw_slice", diag_res["visualizations"].get("raw_mri"))
        seg_mask = diag_res["visualizations"].get("binary_mask", diag_res["visualizations"].get("seg_mask"))
        gradcam = diag_res["visualizations"].get("gradcam_heatmap")

        report_gen = TumorBoardReportGenerator()
        pdf_bytes = report_gen.generate_dossier_pdf(case_data, raw_mri, seg_mask, gradcam)

        safe_filename = f"Tumor_Board_Dossier_{case_data['patient_id']}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF report generation error: {str(e)}")
