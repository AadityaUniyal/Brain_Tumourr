"""
Tests for Phase 6 MLOps Regression Gate and Model Registry.
"""
import json
from pathlib import Path
from scripts.ci_regression_gate import load_metrics, evaluate_gate, GATING_THRESHOLDS

def test_load_metrics_schema():
    metrics = load_metrics()
    assert isinstance(metrics, dict)
    # Check that core keys exist
    for key in ["patient_leakage_rate", "macro_auroc", "overall_accuracy", "selective_accuracy"]:
        assert key in metrics, f"Missing key {key} in extracted metrics"
        assert isinstance(metrics[key], (int, float))

def test_regression_gate_passes_baseline():
    overall_passed, results = evaluate_gate()
    assert overall_passed is True, "CI regression gate failed on current verified benchmark"
    assert len(results) == len(GATING_THRESHOLDS)
    for r in results:
        assert r["passed"] is True, f"Threshold failed for {r['key']}: {r['actual']} vs target {r['target']}"

def test_model_registry_integrity():
    registry_file = Path("models/model_registry.json")
    assert registry_file.exists(), "models/model_registry.json does not exist"
    with open(registry_file) as f:
        registry = json.load(f)
    assert "active_production_models" in registry
    assert len(registry["active_production_models"]) >= 2
    for model in registry["active_production_models"]:
        assert "model_id" in model
        assert "version" in model
        assert "checkpoint_relpath" in model
        assert "status" in model
        assert "Production" in model["status"]
