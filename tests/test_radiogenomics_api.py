"""
Verification test for Radiogenomics and DeepSurv endpoint.
"""
from fastapi.testclient import TestClient
from api.app import app
import numpy as np
import io
from PIL import Image

def test_radiogenomics_endpoint():
    client = TestClient(app)
    
    # 1. Health check
    resp_h = client.get("/health")
    assert resp_h.status_code == 200

    # 2. Synthetic MRI scan
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[32:96, 32:96] = 180
    arr[50:70, 50:70] = 240
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    
    resp = client.post(
        "/api/v1/radiogenomics?patient_age=48.5&patient_sex=MALE",
        files={"file": ("test_scan.png", buf.getvalue(), "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "who_2021_subtype" in data
    assert "idh_mutation_probability" in data
    assert "codeletion_1p19q_probability" in data
    assert "deepsurv_hazard_score" in data
    assert "survival_curve" in data
    assert len(data["survival_curve"]) > 0
    assert "shap_attributions" in data
    assert len(data["shap_attributions"]) > 0
    print("Test passed successfully:", data["who_2021_subtype"])

if __name__ == "__main__":
    test_radiogenomics_endpoint()
