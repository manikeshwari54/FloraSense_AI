"""Image-quality checks and heuristic confidence/OOD prediction decisions.

The module exposes temperature-scaled log-sum-exp energy scoring and threshold
calibration. Its statuses are ``certain``, ``uncertain``, and ``unsuitable``.
These are heuristics calibrated on available validation data, not guarantees
that an input is or is not out of distribution.
"""

import os
import sys
import json
import math
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Default thresholds (empirically calibrated for conservative botanical identification)
DEFAULT_TEMPERATURE = 1.35
DEFAULT_ENERGY_THRESHOLD = -25.0       # Threshold for Helmholtz energy; values lower than this are flagged as unfamiliar
DEFAULT_CONFIDENCE_THRESHOLD = 0.35    # Calibrated minimum top-1 probability for certainty
DEFAULT_MARGIN_THRESHOLD = 0.05        # Minimum gap between rank-1 and rank-2 candidates
BLUR_THRESHOLD = 18.0                  # Laplacian variance threshold (lower values mean blurry/unfocused)
MIN_INTENSITY = 10.0                   # Minimum acceptable mean grayscale brightness (0-255)
MAX_INTENSITY = 248.0                  # Maximum acceptable mean grayscale brightness (0-255)

def check_image_quality(image: Image.Image) -> Tuple[bool, Optional[str]]:
    """Check minimum dimensions, mean brightness, and Laplacian sharpness.

    Returns ``(True, None)`` when checks pass, otherwise ``(False, message)``.
    The implemented minimum dimension is 50 pixels per side.
    """
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    h, w = arr.shape
    if h < 50 or w < 50:
        return False, "Image resolution is too low for botanical identification. Please provide a photo of at least 50x50 pixels."

    mean_intensity = float(np.mean(arr))
    if mean_intensity < MIN_INTENSITY:
        return False, "The image is too dark or underexposed. Please capture with adequate ambient or direct lighting."
    if mean_intensity > MAX_INTENSITY:
        return False, "The image is severely overexposed or washed out. Please capture with balanced lighting."

    # Sample interior patch to compute variance of Laplacian quickly
    interior = arr[1:-1, 1:-1]
    laplacian = (
        arr[0:-2, 1:-1] + arr[2:, 1:-1] +
        arr[1:-1, 0:-2] + arr[1:-1, 2:] - 4 * interior
    )
    laplacian_variance = float(np.var(laplacian))

    if laplacian_variance < BLUR_THRESHOLD:
        return False, (
            f"The image appears too blurry or out of focus (sharpness index: {laplacian_variance:.1f}, "
            f"minimum required: {BLUR_THRESHOLD}). Please upload a sharp photo with clear leaf veins and margins."
        )

    return True, None

def compute_energy_score(logits: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """Return ``T * logsumexp(logits / T)`` for one or more class-score vectors.

    A single vector or a batch of vectors is accepted. Under the current
    rejection rule, scores below the configured threshold are treated as
    unfamiliar; the score is a heuristic, not a calibrated probability.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("Logits must be a class-score vector or a batch of class-score vectors.")
    scaled = logits / temperature
    energy = temperature * torch.logsumexp(scaled, dim=-1)
    return float(energy.mean().item())

def load_thresholds(config_path: str = "./checkpoints/ood_thresholds.json") -> Dict:
    """Load validated thresholds from JSON, using defaults only when absent."""
    defaults = {
        "energy_thresh": DEFAULT_ENERGY_THRESHOLD,
        "conf_thresh": DEFAULT_CONFIDENCE_THRESHOLD,
        "margin_thresh": DEFAULT_MARGIN_THRESHOLD,
        "temperature": DEFAULT_TEMPERATURE,
        "is_calibrated": False,
        "calibration_samples": 0
    }
    if not os.path.exists(config_path):
        return defaults

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thresholds = {
        "energy_thresh": float(data.get("energy_threshold", defaults["energy_thresh"])),
        "conf_thresh": float(data.get("confidence_threshold", defaults["conf_thresh"])),
        "margin_thresh": float(data.get("margin_threshold", defaults["margin_thresh"])),
        "temperature": float(data.get("temperature", defaults["temperature"])),
        "is_calibrated": bool(data.get("is_calibrated", defaults["is_calibrated"])),
        "calibration_samples": int(data.get("calibration_samples", defaults["calibration_samples"])),
    }
    if not all(math.isfinite(thresholds[key]) for key in ("energy_thresh", "conf_thresh", "margin_thresh", "temperature")):
        raise ValueError(f"Threshold configuration contains non-finite values: '{config_path}'")
    if not 0 <= thresholds["conf_thresh"] <= 1 or not 0 <= thresholds["margin_thresh"] <= 1:
        raise ValueError(f"Confidence and margin thresholds must be between 0 and 1: '{config_path}'")
    if thresholds["temperature"] <= 0 or thresholds["calibration_samples"] < 0:
        raise ValueError(f"Temperature and calibration sample count are invalid: '{config_path}'")
    return thresholds

def calibrate_and_filter(
    logits: Optional[torch.Tensor],
    predictions: List[Dict],
    energy_thresh: float = DEFAULT_ENERGY_THRESHOLD,
    conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_thresh: float = DEFAULT_MARGIN_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_ood: bool = True
) -> Dict:
    """Return a status and message for ranked predictions.

    With OOD filtering enabled, predictions are marked uncertain when the energy
    is below its threshold, the top confidence is too low, or the top-two margin
    is too small. With filtering disabled, the confidence threshold is still
    applied. An empty candidate list is unsuitable.
    """
    if not predictions:
        return {
            "status": "unsuitable",
            "reason": "no_predictions",
            "message": "No species candidates could be evaluated.",
            "energy_score": None,
            "predictions": []
        }

    top_cand = predictions[0]
    top_prob = top_cand["confidence"]
    second_prob = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
    margin = top_prob - second_prob

    energy_score = None
    if logits is not None:
        energy_score = round(compute_energy_score(logits, temperature=temperature), 2)

    if not enable_ood:
        is_confident = top_prob >= conf_thresh
        return {
            "status": "certain" if is_confident else "uncertain",
            "reason": "ood_filtering_disabled" if is_confident else "low_confidence",
            "energy_score": energy_score,
            "message": (
                f"Identified as {top_cand['common_name']} (OOD screening disabled)."
                if is_confident
                else f"Prediction is uncertain because top candidate confidence "
                f"({top_cand['score_pct']}%) is below the configured threshold."
            ),
            "predictions": predictions
        }

    # 1. Energy score check (OOD rejection)
    if energy_score is not None and energy_score < energy_thresh:
        return {
            "status": "uncertain",
            "reason": "out_of_distribution",
            "energy_score": energy_score,
            "message": (
                "FloraSense AI could not confidently identify this plant. It appears to belong to an unfamiliar species "
                "not present in this model's trained classes."
            ),
            "predictions": predictions
        }

    # 2. Confidence and margin check
    if top_prob < conf_thresh or margin < margin_thresh:
        return {
            "status": "uncertain",
            "reason": "low_confidence_or_ambiguous",
            "energy_score": energy_score,
            "message": (
                f"Prediction is ambiguous (top candidate confidence: {top_cand['score_pct']}%, margin: {margin*100:.1f}%). "
                "Foliage shares visual patterns with multiple species. Please capture a clearer close-up of leaves and stems."
            ),
            "predictions": predictions
        }

    return {
        "status": "certain",
        "reason": "high_confidence",
        "energy_score": energy_score,
        "message": f"Confirmed with high confidence as {top_cand['common_name']} ({top_cand['scientific_name']}).",
        "predictions": predictions
    }

def calibrate_thresholds_on_validation(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    save_path: str = "./checkpoints/ood_thresholds.json",
    temperature: float = DEFAULT_TEMPERATURE
) -> Dict:
    """Estimate energy/confidence cutoffs from an in-distribution validation loader.

    The energy cutoff is the 5th percentile of all validation energies. The
    confidence and margin cutoffs are the 10th percentile of their corresponding
    values among correctly classified samples (or defaults if none are correct).
    The resulting values are saved as JSON.
    """
    model.eval()
    energies = []
    correct_confidences = []
    correct_margins = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits / temperature, dim=-1)

            # Energy scores
            batch_energies = (temperature * torch.logsumexp(logits / temperature, dim=-1)).cpu().tolist()
            energies.extend(batch_energies)

            # Correct confidences
            top_count = min(2, probs.shape[-1])
            top_probs, top_indices = torch.topk(probs, k=top_count, dim=-1)
            max_probs = top_probs[:, 0]
            preds = top_indices[:, 0]
            margins = max_probs - (top_probs[:, 1] if top_count > 1 else 0)
            for p, margin, pred, lbl in zip(
                max_probs.cpu().tolist(),
                margins.cpu().tolist(),
                preds.cpu().tolist(),
                labels.cpu().tolist(),
            ):
                if pred == lbl:
                    correct_confidences.append(p)
                    correct_margins.append(margin)

    if not energies:
        raise ValueError("Validation loader was empty; cannot calibrate thresholds.")

    calibrated_energy = float(np.percentile(energies, 5))
    calibrated_conf = float(np.percentile(correct_confidences, 10)) if correct_confidences else DEFAULT_CONFIDENCE_THRESHOLD
    calibrated_margin = float(np.percentile(correct_margins, 10)) if correct_margins else DEFAULT_MARGIN_THRESHOLD

    result = {
        "is_calibrated": True,
        "temperature": temperature,
        "energy_threshold": round(calibrated_energy, 2),
        "confidence_threshold": round(calibrated_conf, 4),
        "margin_threshold": round(calibrated_margin, 4),
        "calibration_samples": len(energies),
        "correct_samples": len(correct_confidences),
        "margin_calibration_samples": len(correct_margins),
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
