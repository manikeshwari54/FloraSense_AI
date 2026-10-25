"""
FloraSense AI - Production FastAPI Backend Service
--------------------------------------------------
REST API providing:
  - Real-time indoor plant identification with Out-of-Distribution (OOD) rejection
  - Model switching: EfficientNet-B0 (baseline) or MobileNetV3-Small (lightweight)
  - Image quality and blur screening
  - Botanical profiles for 20 supported indoor plant species
  - Secure decoupling of plant-care profiles from uncertain predictions
  - CORS-enabled for React and frontend clients
"""

import os
import sys
import io
import json
from typing import List, Optional, Dict
from PIL import Image

import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import InferenceEngine, load_inference_engine, resolve_species_metadata
from ood_detector import check_image_quality, calibrate_and_filter, load_thresholds
from plant_data import get_plant_profile, PLANT_DATA

# -------------------------------------------------------------
# FastAPI App Initialization & CORS
# -------------------------------------------------------------
app = FastAPI(
    title="FloraSense AI API",
    description="Indoor plant identification using the class mapping attached to the selected checkpoint.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class PredictionCandidate(BaseModel):
    rank: int
    class_index: int
    raw_label: str
    common_name: str
    scientific_name: str
    confidence: float
    score_pct: float
    model_source: str

class IdentificationResponse(BaseModel):
    status: str = Field(description="'certain' | 'uncertain' | 'unsuitable'")
    reason: Optional[str] = None
    message: str
    energy_score: Optional[float] = None
    predictions: List[PredictionCandidate]
    plant_care: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    device: str
    models_available: List[str]
    active_baseline: str
    num_classes: int
    supported_species: List[str]

# -------------------------------------------------------------
# Model Manager
# -------------------------------------------------------------
class ModelManager:
    _instance = None

    def __init__(self):
        self.device = "CPU"
        self.engines: Dict[str, InferenceEngine] = {}
        configured_dir = os.environ.get("FLORASENSE_CHECKPOINT_DIR", "checkpoints")
        checkpoint_dir = (
            configured_dir
            if os.path.isabs(configured_dir)
            else os.path.join(BASE_DIR, configured_dir)
        )
        self.checkpoint_dir = checkpoint_dir

        # Load available CNN engines
        for arch in ["efficientnet_b0", "mobilenet_v3_small"]:
            eng, err = load_inference_engine(checkpoint_dir, arch)
            if eng is not None:
                self.engines[arch] = eng

        self.num_classes = max((eng.num_classes for eng in self.engines.values()), default=0)
        if self.engines:
            self.device = next(iter(self.engines.values())).get_info()["device"]

    def get_active_engine(self):
        if "efficientnet_b0" in self.engines:
            return self.engines["efficientnet_b0"]
        return next(iter(self.engines.values()), None)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    manager = ModelManager.get_instance()
    available = list(manager.engines.keys())
    active_engine = manager.get_active_engine()
    supported_species = active_engine.class_names if active_engine else []

    return {
        "status": "healthy" if available else "unavailable",
        "device": manager.device,
        "models_available": available,
        "active_baseline": "efficientnet_b0" if "efficientnet_b0" in manager.engines else (available[0] if available else "none"),
        "num_classes": manager.num_classes,
        "supported_species": supported_species
    }

@app.get("/api/v1/species")
def list_species():
    manager = ModelManager.get_instance()
    engine = manager.get_active_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No usable checkpoint is available.",
        )
    return {
        "total_classes": engine.num_classes,
        "species": [
            {
                "id": index,
                "name": label,
                "common_name": resolve_species_metadata(label)[1],
                "scientific_name": resolve_species_metadata(label)[0],
                "care_available": get_plant_profile(label) is not None,
            }
            for index, label in enumerate(engine.class_names)
        ]
    }

@app.get("/api/v1/species/{species_name}")
def get_species_care(species_name: str):
    profile = get_plant_profile(species_name)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Care profile for '{species_name}' not found."
        )
    return profile

@app.post("/api/v1/identify", response_model=IdentificationResponse)
async def identify_plant(
    file: UploadFile = File(...),
    arch: str = Query("efficientnet_b0", enum=["efficientnet_b0", "mobilenet_v3_small"]),
    top_k: int = Query(5, ge=1, le=5),
    energy_thresh: Optional[float] = Query(None, description="Override the checkpoint energy threshold"),
    conf_thresh: Optional[float] = Query(None, ge=0, le=1, description="Override the checkpoint confidence threshold"),
    margin_thresh: Optional[float] = Query(None, ge=0, le=1, description="Override the checkpoint top-two margin threshold"),
    temperature: Optional[float] = Query(None, gt=0, description="Override the checkpoint energy temperature"),
    enable_ood: bool = Query(True, description="Enable energy-based OOD rejection")
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Allowed formats: JPEG, PNG, WebP."
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 10 MB limit."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image: {str(e)}"
        )

    # Image Quality Pre-Check
    is_valid_quality, quality_error = check_image_quality(image)
    if not is_valid_quality:
        return {
            "status": "unsuitable",
            "reason": "image_quality_failed",
            "message": quality_error,
            "energy_score": None,
            "predictions": [],
            "plant_care": None
        }

    manager = ModelManager.get_instance()
    engine = manager.engines.get(arch)
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model architecture '{arch}' is not initialized or checkpoint is missing."
        )

    thresholds = load_thresholds(
        os.path.join(manager.checkpoint_dir, "ood_thresholds.json")
    )
    logits, predictions = engine.predict(image, top_k=top_k)

    # Apply OOD and Confidence Decision Boundary
    decision = calibrate_and_filter(
        logits=logits,
        predictions=predictions,
        energy_thresh=thresholds["energy_thresh"] if energy_thresh is None else energy_thresh,
        conf_thresh=thresholds["conf_thresh"] if conf_thresh is None else conf_thresh,
        margin_thresh=thresholds["margin_thresh"] if margin_thresh is None else margin_thresh,
        temperature=thresholds["temperature"] if temperature is None else temperature,
        enable_ood=enable_ood
    )

    plant_care = None
    if decision["status"] == "certain" and decision["predictions"]:
        top_name = decision["predictions"][0]["common_name"]
        plant_care = get_plant_profile(top_name)

    return {
        "status": decision["status"],
        "reason": decision.get("reason"),
        "message": decision["message"],
        "energy_score": decision.get("energy_score"),
        "predictions": decision["predictions"],
        "plant_care": plant_care
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
