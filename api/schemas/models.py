"""
Pydantic Request & Response Schemas for Medical Diagnostics and Radiogenomics.
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


class SurvivalCurvePoint(BaseModel):
    month: float
    survival_prob: float


class RadiogenomicsResponse(BaseModel):
    status: str
    filename: str
    patient_age: float
    patient_sex: str
    idh_mutation_probability: float
    idh_status: str
    idh_clinical_guidance: str
    codeletion_1p19q_probability: float
    codeletion_1p19q_status: str
    who_2021_subtype: str
    deepsurv_hazard_score: float
    risk_tier: str
    median_overall_survival_months: float
    survival_curve: List[SurvivalCurvePoint]
    radiomic_summary: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    gpu_available: bool
