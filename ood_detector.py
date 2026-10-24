"""
FloraSense AI - Out-Of-Distribution (OOD) & Safe Rejection Engine
-----------------------------------------------------------------
Features:
  1. Image Quality / Blur Pre-Check (Laplacian variance & intensity bounds)
  2. Temperature-Scaled Helmholtz Free Energy Scoring
  3. Calibrated Multi-factor Decision Boundary:
       - 'certain': High confidence known in-distribution species
       - 'uncertain': Low confidence, ambiguous margin, or out-of-distribution
       - 'unsuitable': Blurry, underexposed, or corrupt image input
  4. Configurable threshold persistence and empirical calibration routine
"""

import os
import sys
import json
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
    """
    Evaluates image quality, focus/blur, and exposure lighting.
    Returns (is_acceptable: bool, error_message: str or None).
    """
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    h, w = arr.shape
    if h < 50 or w < 50:
        return False, "Image resolution is too low for botanical identification. Please provide a photo of at least 224x224 pixels."

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
    """
    Computes Helmholtz free energy score:
      S_energy(x) = T * log(sum(exp(logits / T)))
    Higher energy score indicates in-distribution confidence; lower indicates unfamiliar/OOD.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    scaled = logits / temperature
    energy = temperature * torch.logsumexp(scaled, dim=-1)
    return float(energy.mean().item())

def load_thresholds(config_path: str = "./checkpoints/ood_thresholds.json") -> Dict:
    """
    Loads saved calibration thresholds, or falls back to robust defaults.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "energy_thresh": float(data.get("energy_threshold", DEFAULT_ENERGY_THRESHOLD)),
                "conf_thresh": float(data.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)),
                "margin_thresh": float(data.get("margin_threshold", DEFAULT_MARGIN_THRESHOLD)),
                "temperature": float(data.get("temperature", DEFAULT_TEMPERATURE)),
                "is_calibrated": data.get("is_calibrated", False),
                "calibration_samples": data.get("calibration_samples", 0)
            }
        except Exception:
            pass

    return {
        "energy_thresh": DEFAULT_ENERGY_THRESHOLD,
        "conf_thresh": DEFAULT_CONFIDENCE_THRESHOLD,
        "margin_thresh": DEFAULT_MARGIN_THRESHOLD,
        "temperature": DEFAULT_TEMPERATURE,
        "is_calibrated": False,
        "calibration_samples": 0
    }

def calibrate_and_filter(
    logits: Optional[torch.Tensor],
    predictions: List[Dict],
    energy_thresh: float = DEFAULT_ENERGY_THRESHOLD,
    conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_thresh: float = DEFAULT_MARGIN_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_ood: bool = True
) -> Dict:
    """
    Applies calibrated energy score and confidence thresholding to establish
    whether a prediction is 'certain' or 'uncertain'.
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
        try:
            energy_score = round(compute_energy_score(logits, temperature=temperature), 2)
        except Exception:
            energy_score = None

    if not enable_ood:
        return {
            "status": "certain",
            "reason": "ood_filtering_disabled",
            "energy_score": energy_score,
            "message": f"Identified as {top_cand['common_name']} (OOD screening disabled).",
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
    """
    Calibrates energy and confidence thresholds from in-distribution validation data.
    Sets energy threshold at the 5th percentile of in-distribution scores (allowing 95% through),
    and confidence threshold at the 10th percentile of correct predictions.
    """
    model.eval()
    energies = []
    correct_confidences = []

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
            max_probs, preds = torch.max(probs, dim=-1)
            for p, pred, lbl in zip(max_probs.cpu().tolist(), preds.cpu().tolist(), labels.cpu().tolist()):
                if pred == lbl:
                    correct_confidences.append(p)

    if not energies:
        raise ValueError("Validation loader was empty; cannot calibrate thresholds.")

    calibrated_energy = float(np.percentile(energies, 5))
    calibrated_conf = float(np.percentile(correct_confidences, 10)) if correct_confidences else DEFAULT_CONFIDENCE_THRESHOLD

    result = {
        "is_calibrated": True,
        "temperature": temperature,
        "energy_threshold": round(calibrated_energy, 2),
        "confidence_threshold": round(calibrated_conf, 4),
        "margin_threshold": DEFAULT_MARGIN_THRESHOLD,
        "calibration_samples": len(energies),
        "correct_samples": len(correct_confidences)
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
