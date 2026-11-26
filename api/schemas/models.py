"""
Pydantic Request & Response Schemas for Medical Diagnostics.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClassProbabilities(BaseModel):
    Meningioma: float = Field(..., description="Calibrated probability of Meningioma")
    Glioma: float = Field(..., description="Calibrated probability of Glioma")
    Pituitary: float = Field(..., description="Calibrated probability of Pituitary tumor")


class DiagnosticDetails(BaseModel):
    predicted_tumor_type: str
    confidence: float
    class_probabilities: ClassProbabilities
    conformal_prediction_set: List[str] = Field(..., description="Guaranteed 95% coverage prediction set")
    is_clinically_uncertain: bool
    coverage_guarantee: float = 0.95


class TumorQuantification(BaseModel):
    tumor_pixel_count: int
    tumor_area_mm2: float
    estimated_volume_cm3: float


class DiagnosticResponse(BaseModel):
    status: str
    filename: str
    is_ood: bool
    ood_message: str
    energy_score: float
    diagnostic: DiagnosticDetails
    quantification: TumorQuantification
    gradcam_base64: Optional[str] = None
    mask_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    gpu_available: bool
