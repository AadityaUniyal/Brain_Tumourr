"""
Clinical Diagnostic Endpoint:
Handles DICOM, NIfTI, MAT, and standard image formats.
Runs multi-task segmentation, classification, OOD verification, and Grad-CAM generation.
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from api.schemas.models import DiagnosticResponse
from src.inference.predictor import ClinicalPredictor

router = APIRouter(prefix="/api/v1", tags=["Diagnostic Inference"])

# Global singleton predictor instance
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ClinicalPredictor()
    return _predictor


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
            gradcam_base64=cam_overlay_b64,
            mask_base64=mask_b64,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic pipeline error: {str(e)}")
