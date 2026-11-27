"""
Integration tests for FastAPI endpoints using TestClient.
"""
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_diagnose_endpoint_with_synthetic_mri():
    # Generate synthetic 128x128 grayscale MRI image
    img = Image.fromarray(np.random.randint(20, 200, (128, 128), dtype=np.uint8))
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    buff.seek(0)

    response = client.post(
        "/api/v1/diagnose",
        files={"file": ("test_mri.jpg", buff, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "diagnostic" in data
    assert "quantification" in data
    assert "gradcam_base64" in data
    assert "selective_triage" in data
    assert data["selective_triage"]["action"] in ["AUTOMATED_DIAGNOSIS", "NEUROPATHOLOGIST_TRIAGE"]
    assert data["diagnostic"]["predicted_tumor_type"] in ["Meningioma", "Glioma", "Pituitary"]
