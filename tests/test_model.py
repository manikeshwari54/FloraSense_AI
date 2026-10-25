import os
import json
import pytest
import torch
from model import (
    InferenceEngine,
    load_inference_engine,
    resolve_species_metadata
)
from train_models import build_model

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

def test_47_class_checkpoint_mapping_and_prediction_labels(tmp_path, sample_plant_image):
    class_names = [f"Plant {index:02d}" for index in range(47)]
    id2label = {str(index): name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in enumerate(class_names)}
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({"id2label": id2label, "label2id": label2id}),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "efficientnet_b0_best.pt"
    model = build_model("efficientnet_b0", num_classes=len(class_names), pretrained=False)
    torch.save(
        {
            "arch": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "best_acc": 0.25,
        },
        checkpoint_path,
    )

    engine = InferenceEngine(
        arch="efficientnet_b0",
        checkpoint_path=str(checkpoint_path),
        class_mapping_path=str(mapping_path),
        device="cpu",
    )
    logits, predictions = engine.predict(sample_plant_image, top_k=5)

    assert logits.shape == (1, 47)
    assert engine.num_classes == 47
    assert all(
        engine.id2label[prediction["class_index"]] == prediction["raw_label"]
        for prediction in predictions
    )
    assert all(prediction["common_name"] in class_names for prediction in predictions)

def test_inference_engine_rejects_mismatched_mapping(tmp_path):
    checkpoint_path = tmp_path / "efficientnet_b0_best.pt"
    mapping_path = tmp_path / "class_mapping.json"
    mapping_path.write_text(
        json.dumps({
            "id2label": {"0": "Plant A", "1": "Plant B"},
            "label2id": {"Plant A": 1, "Plant B": 0},
        }),
        encoding="utf-8",
    )
    checkpoint_path.touch()

    with pytest.raises(ValueError, match="do not match"):
        InferenceEngine(
            arch="efficientnet_b0",
            checkpoint_path=str(checkpoint_path),
            class_mapping_path=str(mapping_path),
        )
