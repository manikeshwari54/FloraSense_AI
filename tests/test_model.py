import os
import pytest
import torch
from model import (
    InferenceEngine,
    load_inference_engine,
    resolve_species_metadata
)

def test_inference_engine_list_available(checkpoint_dir):
    avail = InferenceEngine.list_available(checkpoint_dir)
    assert isinstance(avail, list)
    for entry in avail:
        assert "arch" in entry
        assert "checkpoint_exists" in entry
        assert "is_usable" in entry

def test_inference_engine_loads(checkpoint_dir):
    engine, err = load_inference_engine(checkpoint_dir, "efficientnet_b0")
    if engine is not None:
        info = engine.get_info()
        assert info["arch"] == "efficientnet_b0"
        assert info["num_classes"] > 0
    else:
        # Checkpoint might not be trained yet
        assert err is not None

def test_inference_engine_predict_shape(checkpoint_dir, sample_plant_image):
    engine, err = load_inference_engine(checkpoint_dir, "efficientnet_b0")
    if engine is None:
        pytest.skip("EfficientNet-B0 checkpoint not found on disk")

    logits, predictions = engine.predict(sample_plant_image, top_k=5)
    assert logits.shape[1] == engine.num_classes
    assert len(predictions) <= 5
    assert len(predictions) > 0

    top = predictions[0]
    required_keys = ["rank", "class_index", "raw_label", "common_name", "scientific_name", "confidence", "model_source"]
    for k in required_keys:
        assert k in top

def test_inference_engine_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        InferenceEngine(arch="efficientnet_b0", checkpoint_path="./nonexistent_ckpt_12345.pt")

def test_load_inference_engine_returns_tuple():
    engine, err = load_inference_engine("./nonexistent_dir", "efficientnet_b0")
    assert engine is None
    assert isinstance(err, str)

def test_resolve_species_metadata_snake_plant():
    sci, com = resolve_species_metadata("Snake plant")
    assert "Snake" in sci or "Snake" in com
