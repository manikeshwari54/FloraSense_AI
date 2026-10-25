import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ood_detector import (
    check_image_quality,
    compute_energy_score,
    calibrate_and_filter,
    load_thresholds,
    calibrate_thresholds_on_validation,
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

def test_calibrated_thresholds_load_from_checkpoint_config(tmp_path):
    class FixedLogitModel(nn.Module):
        def forward(self, images):
            logits = torch.zeros((images.shape[0], 47))
            logits[:, 0] = 5.0
            logits[:, 1] = 4.0
            return logits

    loader = DataLoader(
        TensorDataset(torch.zeros((4, 3, 8, 8)), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    config_path = tmp_path / "ood_thresholds.json"
    result = calibrate_thresholds_on_validation(
        FixedLogitModel(),
        loader,
        device=torch.device("cpu"),
        save_path=str(config_path),
        temperature=1.2,
    )
    loaded = load_thresholds(str(config_path))

    assert result["calibration_samples"] == 4
    assert result["margin_calibration_samples"] == 4
    assert result["margin_threshold"] > 0
    assert loaded["energy_thresh"] == result["energy_threshold"]
    assert loaded["temperature"] == 1.2
    assert loaded["margin_thresh"] == result["margin_threshold"]

def test_calibrate_and_filter_uses_top_two_margin():
    predictions = [
        {"common_name": "Plant A", "scientific_name": "Plant A", "confidence": 0.8, "score_pct": 80.0},
        {"common_name": "Plant B", "scientific_name": "Plant B", "confidence": 0.78, "score_pct": 78.0},
    ]
    result = calibrate_and_filter(
        logits=torch.tensor([[10.0, 9.0]]),
        predictions=predictions,
        energy_thresh=-100.0,
        conf_thresh=0.5,
        margin_thresh=0.05,
    )

    assert result["status"] == "uncertain"
    assert result["reason"] == "low_confidence_or_ambiguous"
