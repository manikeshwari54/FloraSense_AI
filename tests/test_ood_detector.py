import pytest
import numpy as np
from PIL import Image
import torch
from ood_detector import (
    check_image_quality,
    compute_energy_score,
    calibrate_and_filter,
    load_thresholds
)

def test_quality_ok_noisy_image(sample_plant_image):
    ok, err = check_image_quality(sample_plant_image)
    assert ok is True
    assert err is None

def test_quality_rejects_solid_color():
    solid = Image.new("RGB", (224, 224), color=(80, 140, 90))
    ok, err = check_image_quality(solid)
    assert ok is False
    assert "blurry" in err.lower() or "focus" in err.lower()

def test_quality_rejects_tiny():
    tiny = Image.new("RGB", (32, 32), color=(80, 140, 90))
    ok, err = check_image_quality(tiny)
    assert ok is False
    assert "resolution" in err.lower()

def test_energy_score_type():
    logits = torch.tensor([[2.5, 0.1, -1.2, 3.4]])
    energy = compute_energy_score(logits)
    assert isinstance(energy, float)

def test_calibrate_returns_status():
    predictions = [
        {"common_name": "Tulsi", "scientific_name": "Ocimum tenuiflorum", "confidence": 0.85, "score_pct": 85.0},
        {"common_name": "Mint", "scientific_name": "Mentha spicata", "confidence": 0.08, "score_pct": 8.0}
    ]
    logits = torch.tensor([[4.5, 1.0, -1.0]])
    res = calibrate_and_filter(logits, predictions, conf_thresh=0.5, energy_thresh=-50.0)
    assert res["status"] in ["certain", "uncertain", "unsuitable"]
    assert len(res["predictions"]) == 2

def test_calibrate_uncertain_on_low_confidence():
    predictions = [
        {"common_name": "Tulsi", "scientific_name": "Ocimum tenuiflorum", "confidence": 0.25, "score_pct": 25.0},
        {"common_name": "Mint", "scientific_name": "Mentha spicata", "confidence": 0.24, "score_pct": 24.0}
    ]
    logits = torch.tensor([[0.5, 0.48, 0.2]])
    res = calibrate_and_filter(logits, predictions, conf_thresh=0.5)
    assert res["status"] == "uncertain"
    assert "low_confidence" in res["reason"] or "ambiguous" in res["reason"]

def test_load_thresholds_defaults():
    t = load_thresholds("./nonexistent_path_xyz.json")
    assert "energy_thresh" in t
    assert "conf_thresh" in t
    assert isinstance(t["energy_thresh"], float)
    assert isinstance(t["conf_thresh"], float)
