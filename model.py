"""
FloraSense AI — Indoor Plant Identification Engine
---------------------------------------------------
Provides InferenceEngine: Production loader and predictor for trained CNN checkpoints
(EfficientNet-B0, MobileNetV3-Small) bound to their specific class mappings.
Trained on the House Plant Species dataset (kakasher/house-plant-species, CC BY 3.0).
20 supported indoor plant classes.
"""

import os
import sys
import io
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import SPECIES_TAXONOMY, IMAGENET_MEAN, IMAGENET_STD

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Canonical botanical aliases for robust taxonomy lookup without altering class index predictions
CANONICAL_ALIASES = {
    "snake plant": {"common_name": "Snake Plant / Mother-in-Law's Tongue", "scientific_name": "Dracaena trifasciata (syn. Sansevieria trifasciata)"},
    "sansevieria": {"common_name": "Snake Plant / Mother-in-Law's Tongue", "scientific_name": "Dracaena trifasciata (syn. Sansevieria trifasciata)"},
    "pothos": {"common_name": "Golden Pothos / Devil's Ivy", "scientific_name": "Epipremnum aureum"},
}

def resolve_species_metadata(label: str) -> Tuple[str, str]:
    """
    Resolves canonical common name and scientific name from a predicted model label.
    Never fabricates a class; only formats and normalizes display strings.
    """
    clean = label.replace("_", " ").strip()
    norm_key = clean.lower()

    # 1. Check exact canonical aliases
    if norm_key in CANONICAL_ALIASES:
        meta = CANONICAL_ALIASES[norm_key]
        return meta["scientific_name"], meta["common_name"]

    # 2. Check SPECIES_TAXONOMY
    for key, tax in SPECIES_TAXONOMY.items():
        if key.lower() == norm_key:
            return tax["scientific_name"], tax["common_name"]

    # 3. Parse parentheses if label is formatted like 'Common Name (Scientific Name)'
    if "(" in clean and ")" in clean:
        parts = clean.split("(")
        common = parts[0].strip()
        sci = parts[1].split(")")[0].strip()
        return sci, common

    return clean, clean

class InferenceEngine:
    """
    Unified production inference engine for PyTorch transfer-learning models.
    Ties the model weights strictly to the specific class mapping it was trained with.
    """
    def __init__(
        self,
        arch: str,
        checkpoint_path: str,
        class_mapping_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.arch = arch
        self.checkpoint_path = checkpoint_path

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint_path}'")

        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if class_mapping_path is None:
            class_mapping_path = os.path.join(ckpt_dir, "class_mapping.json")

        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(
                f"Class mapping file missing at '{class_mapping_path}'. "
                f"Each checkpoint must be accompanied by its class_mapping.json to prevent class index mismatches."
            )

        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        raw_id2label = mapping_data.get("id2label", {})
        # Ensure integer keys
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.class_names = [self.id2label[i] for i in range(len(self.id2label))]
        self.num_classes = len(self.class_names)

        # Build model and load state dict
        from train_models import build_model
        self.model = build_model(arch, num_classes=self.num_classes, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.best_acc = checkpoint.get("best_acc", None)
        self.epoch = checkpoint.get("epoch", None)

        # Standard inference transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image: Image.Image, top_k: int = 5) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Runs inference on a PIL image.
        Returns:
          logits: raw unscaled torch.Tensor (1, num_classes) on CPU
          predictions: ranked list of top_k prediction dictionaries
        """
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)[0]

        k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=k)

        results = []
        for rank, (p, idx) in enumerate(zip(top_probs, top_indices), 1):
            class_idx = idx.item()
            raw_label = self.id2label.get(class_idx, f"Class_{class_idx}")
            sci_name, com_name = resolve_species_metadata(raw_label)
            confidence = float(p.item())

            results.append({
                "rank": rank,
                "class_index": class_idx,
                "raw_label": raw_label,
                "common_name": com_name,
                "scientific_name": sci_name,
                "confidence": round(confidence, 4),
                "score_pct": round(confidence * 100, 2),
                "model_source": f"{self.arch.upper()} (Checkpoint)"
            })

        return logits.cpu(), results

    def get_info(self) -> Dict:
        return {
            "arch": self.arch,
            "device": str(self.device).upper(),
            "num_classes": self.num_classes,
            "best_acc": self.best_acc,
            "epoch": self.epoch,
            "checkpoint_path": self.checkpoint_path
        }

    @staticmethod
    def list_available(checkpoint_dir: str = "./checkpoints") -> List[Dict]:
        """
        Scans directory and returns metadata diagnostics for all checkpoints.
        Does not crash if individual files are corrupt or incomplete.
        """
        results = []
        if not os.path.exists(checkpoint_dir):
            return results

        arches = ["efficientnet_b0", "mobilenet_v3_small"]
        for arch in arches:
            ckpt_path = os.path.join(checkpoint_dir, f"{arch}_best.pt")
            mapping_path = os.path.join(checkpoint_dir, "class_mapping.json")

            entry = {
                "arch": arch,
                "checkpoint_path": ckpt_path,
                "checkpoint_exists": os.path.exists(ckpt_path),
                "mapping_exists": os.path.exists(mapping_path),
                "is_usable": False,
                "num_classes": 0,
                "best_acc": None,
                "error": None
            }

            if entry["checkpoint_exists"] and entry["mapping_exists"]:
                try:
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        mdata = json.load(f)
                    classes = list(mdata.get("id2label", {}).values())
                    entry["num_classes"] = len(classes)

                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    entry["best_acc"] = ckpt.get("best_acc")
                    entry["is_usable"] = True
                except Exception as e:
                    entry["error"] = str(e)
            results.append(entry)

        return results

def load_inference_engine(checkpoint_dir: str = "./checkpoints", arch: str = "efficientnet_b0") -> Tuple[Optional[InferenceEngine], Optional[str]]:
    """
    Safely instantiates an InferenceEngine. Returns (engine, error_message).
    """
    ckpt_path = os.path.join(checkpoint_dir, f"{arch}_best.pt")
    mapping_path = os.path.join(checkpoint_dir, "class_mapping.json")

    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found at '{ckpt_path}'"
    if not os.path.exists(mapping_path):
        return None, f"Class mapping not found at '{mapping_path}'"

    try:
        engine = InferenceEngine(arch=arch, checkpoint_path=ckpt_path, class_mapping_path=mapping_path)
        return engine, None
    except Exception as e:
        return None, f"Failed to initialize {arch}: {str(e)}"
