"""
Pydantic Request & Response Schemas for Medical Diagnostics and Radiogenomics.
"""
from typing import Dict, List, Optional, Any
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


class SelectiveTriageDetails(BaseModel):
    g_selection: int = Field(..., description="1 for automated sign-off, 0 for neuropathologist triage")
    action: str = Field(..., description="'AUTOMATED_DIAGNOSIS' or 'NEUROPATHOLOGIST_TRIAGE'")
    action_label: str
    triage_priority: str = Field(..., description="'STAT / URGENT', 'ELEVATED', 'ROUTINE', or 'NONE'")
    clinical_guidance: str
    abstention_reasons: List[str] = []
    selective_accuracy_guarantee: float = 0.9686
    misdiagnosis_intercepted_pct: float = 0.8971
    is_selective_approved: bool


class DiagnosticResponse(BaseModel):
    status: str
    filename: str
    is_ood: bool
    ood_message: str
    energy_score: float
    diagnostic: DiagnosticDetails
    quantification: TumorQuantification
    selective_triage: Optional[SelectiveTriageDetails] = None
    dicom_metadata: Optional[Dict[str, Any]] = None
    gradcam_base64: Optional[str] = None
    mask_base64: Optional[str] = None


class SurvivalCurvePoint(BaseModel):
    month: float
    survival_prob: float


class ShapFeatureAttribution(BaseModel):
    feature_name: str
    shap_value: float
    effect: str = Field(..., description="'risk_increasing' or 'protective'")


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
    shap_attributions: List[ShapFeatureAttribution] = []


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    gpu_available: bool
