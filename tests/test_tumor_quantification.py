"""
Unit tests for Tumor Volumetrics and RANO Bidirectional Calliper Metrology.
"""
import pytest
import numpy as np
from src.inference.tumor_quantification import compute_rano_callipers


def test_empty_mask_rano():
    empty_mask = np.zeros((128, 128), dtype=np.uint8)
    res = compute_rano_callipers(empty_mask, pixel_spacing_mm=0.8)
    assert res["major_axis_mm"] == 0.0
    assert res["minor_axis_mm"] == 0.0
    assert res["product_cm2"] == 0.0
    assert len(res["major_line"]) == 0
    assert len(res["minor_line"]) == 0


def test_synthetic_rectangle_rano():
    # Create a 40x20 pixel rectangular lesion centered at (64, 64)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[54:74, 44:84] = 1  # height 20, width 40

    spacing = 1.0  # 1 mm per pixel
    res = compute_rano_callipers(mask, pixel_spacing_mm=spacing)

    # Major axis should be ~40mm, minor ~20mm
    assert res["major_axis_mm"] >= res["minor_axis_mm"]
    assert 38.0 <= res["major_axis_mm"] <= 42.0
    assert 18.0 <= res["minor_axis_mm"] <= 22.0
    assert res["product_cm2"] > 0.0

    # Verify calliper endpoints
    assert len(res["major_line"]) == 2
    assert len(res["minor_line"]) == 2

    # Check perpendicularity: dot product of directional vectors should be close to 0
    p1, p2 = np.array(res["major_line"][0]), np.array(res["major_line"][1])
    q1, q2 = np.array(res["minor_line"][0]), np.array(res["minor_line"][1])

    v_major = p2 - p1
    v_minor = q2 - q1

    cos_sim = np.abs(np.dot(v_major, v_minor) / (np.linalg.norm(v_major) * np.linalg.norm(v_minor) + 1e-6))
    assert cos_sim < 0.05, f"Callipers should be orthogonal, but cos_sim={cos_sim}"


def test_clinical_predictor_rano_integration():
    from src.inference.predictor import ClinicalPredictor
    predictor = ClinicalPredictor()
    
    # Create synthetic test image
    synthetic_img = (np.random.rand(128, 128) * 255).astype(np.uint8)
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(synthetic_img).save(buf, format="PNG")
    
    res = predictor.predict(buf.getvalue(), filename="test_scan.png")
    assert "quantification" in res
    assert "rano_callipers" in res["quantification"]
    assert "major_axis_mm" in res["quantification"]["rano_callipers"]
    assert "minor_axis_mm" in res["quantification"]["rano_callipers"]
    assert "product_cm2" in res["quantification"]["rano_callipers"]
