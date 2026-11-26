"""
Health and telemetry endpoints.
"""
import torch
from fastapi import APIRouter
from api.schemas.models import HealthResponse

router = APIRouter(tags=["System Health"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="healthy",
        app_name="NeuroVision Elite",
        version="2.0.0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu_available=torch.cuda.is_available(),
    )
