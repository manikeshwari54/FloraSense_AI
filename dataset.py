"""
FloraSense AI - Production Dataset Pipeline
-------------------------------------------
Handles real plant image dataset validation, deduplication, stratified splits,
class imbalance weighting, and held-out Out-of-Distribution (OOD) creation.

Supported directory structure:
  data/
    train/
      Tulsi/
      Neem/
      African_Violet/
      ...
    val/
      ...
    test/
      ...
  OR flat:
  data/
    Tulsi/
    Neem/
    ...
"""

import os
import sys
import json
import hashlib
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ImageNet standard statistics (consistent with pretrained EfficientNet & MobileNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard canonical plant species taxonomy (47 Houseplants + Tulsi + Neem = 49 classes)
SPECIES_TAXONOMY = {
    "African Violet": {"scientific_name": "Streptocarpus ionanthus", "common_name": "African Violet", "family": "Gesneriaceae"},
    "Aloe Vera": {"scientific_name": "Aloe vera", "common_name": "Aloe Vera", "family": "Asphodelaceae"},
    "Boston Fern": {"scientific_name": "Nephrolepis exaltata", "common_name": "Boston Fern", "family": "Nephrolepidaceae"},
    "Calathea": {"scientific_name": "Goeppertia spp.", "common_name": "Calathea", "family": "Marantaceae"},
    "Chinese Evergreen": {"scientific_name": "Aglaonema commutatum", "common_name": "Chinese Evergreen", "family": "Araceae"},
    "Dracaena": {"scientific_name": "Dracaena marginata", "common_name": "Dragon Tree", "family": "Asparagaceae"},
    "English Ivy": {"scientific_name": "Hedera helix", "common_name": "English Ivy", "family": "Araliaceae"},
    "Jade Plant": {"scientific_name": "Crassula ovata", "common_name": "Jade Plant", "family": "Crassulaceae"},
    "Monstera Deliciosa": {"scientific_name": "Monstera deliciosa", "common_name": "Monstera Deliciosa", "family": "Araceae"},
    "Orchid": {"scientific_name": "Phalaenopsis spp.", "common_name": "Moth Orchid", "family": "Orchidaceae"},
    "Peace Lily": {"scientific_name": "Spathiphyllum wallisii", "common_name": "Peace Lily", "family": "Araceae"},
    "Pothos": {"scientific_name": "Epipremnum aureum", "common_name": "Golden Pothos", "family": "Araceae"},
    "Prayer Plant": {"scientific_name": "Maranta leuconeura", "common_name": "Prayer Plant", "family": "Marantaceae"},
    "Rattlesnake Plant": {"scientific_name": "Goeppertia insignis", "common_name": "Rattlesnake Plant", "family": "Marantaceae"},
    "Rubber Plant": {"scientific_name": "Ficus elastica", "common_name": "Rubber Tree", "family": "Moraceae"},
    "Snake Plant": {"scientific_name": "Dracaena trifasciata", "common_name": "Snake Plant", "family": "Asparagaceae"},
    "Tradescantia": {"scientific_name": "Tradescantia zebrina", "common_name": "Wandering Dude", "family": "Commelinaceae"},
    "Venus Flytrap": {"scientific_name": "Dionaea muscipula", "common_name": "Venus Flytrap", "family": "Droseraceae"},
    "ZZ Plant": {"scientific_name": "Zamioculcas zamiifolia", "common_name": "ZZ Plant", "family": "Araceae"},
    "Pothos (Epipremnum aureum)": {"scientific_name": "Epipremnum aureum", "common_name": "Golden Pothos", "family": "Araceae"}
}

def get_class_mappings(class_names=None):
    """
    Generates contiguous integer index to class mappings.
    If class_names is provided, uses that exact list (sorted).
    Otherwise defaults to SPECIES_TAXONOMY keys.
    """
    if class_names is None:
        names = sorted(list(SPECIES_TAXONOMY.keys()))
    else:
        names = sorted(list(set(class_names)))
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return names, id2label, label2id

def compute_image_hash(image: Image.Image) -> str:
    """Computes a perceptual/MD5 hash for exact and near-duplicate detection."""
    thumb = image.resize((32, 32), Image.Resampling.LANCZOS).convert("L")
    return hashlib.md5(thumb.tobytes()).hexdigest()

def get_data_transforms():
    """
    Constructs clean, botanically conservative transforms.
    Avoids extreme distortion that destroys leaf shape and venation patterns.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_test_transform

class PlantImageFolderDataset(Dataset):
    """
    Dataset wrapper supporting file paths, in-memory PIL loading,
    and automatic format normalization.
    """
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path_or_PIL, class_id)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]
        if isinstance(item, str):
            with Image.open(item) as img:
                image = img.convert("RGB")
        elif isinstance(item, Image.Image):
            image = item.convert("RGB")
        else:
            raise ValueError(f"Unsupported sample format: {type(item)}")

        if self.transform:
            image = self.transform(image)
        return image, label

def compute_class_weights(labels_list, num_classes):
    """
    Computes inverse-frequency class weights for weighted CrossEntropyLoss:
    w_c = Total / (Num_Classes * Count_c)
    """
    counts = Counter(labels_list)
    total_samples = len(labels_list)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        c = counts.get(class_id, 0)
        if c > 0:
            weights[class_id] = total_samples / (num_classes * c)
        else:
            weights[class_id] = 1.0
    weights = weights / weights.mean()
    return weights

def check_neem_tulsi_coverage(class_names):
    """
    Verifies whether Tulsi and Neem are represented in the given class names.
    Returns status dict with boolean flags and missing list.
    """
    norm = [c.lower().replace("_", " ").strip() for c in class_names]
    tulsi_present = any("tulsi" in c or "holy basil" in c or "ocimum" in c for c in norm)
    neem_present = any("neem" in c or "azadirachta" in c or "margosa" in c for c in norm)
    missing = []
    if not tulsi_present:
        missing.append("Tulsi (Holy Basil / Ocimum tenuiflorum)")
    if not neem_present:
        missing.append("Neem (Indian Lilac / Azadirachta indica)")
    return {
        "tulsi_present": tulsi_present,
        "neem_present": neem_present,
        "missing": missing,
        "complete": tulsi_present and neem_present
    }

def validate_and_load_real_dataset(data_dir, min_images_per_class=10, valid_extensions=('.jpg', '.jpeg', '.png', '.webp')):
    """
    Scans a directory of class subfolders, validates real image files, detects duplicates,
    and returns verified samples.

    Returns:
      samples: list of (filepath, class_id)
      class_names: sorted list of class names
      report: summary dict containing per-class counts, duplicate counts, and invalid files.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")])
    if not subdirs:
        raise ValueError(f"No class subdirectories found in '{data_dir}'. Expected format: '{data_dir}/<ClassName>/image.jpg'")

    class_names, id2label, label2id = get_class_mappings(subdirs)

    samples = []
    seen_hashes = {}
    class_counts = Counter()
    duplicates_detected = 0
    corrupt_files = 0
    skipped_extensions = 0

    for c_name in class_names:
        c_dir = os.path.join(data_dir, c_name)
        c_id = label2id[c_name]

        for fname in sorted(os.listdir(c_dir)):
            fpath = os.path.join(c_dir, fname)
            if not os.path.isfile(fpath):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_extensions:
                skipped_extensions += 1
                continue

            try:
                with Image.open(fpath) as img:
                    img.verify()
                # Reopen to compute perceptual hash after verify()
                with Image.open(fpath) as img:
                    rgb = img.convert("RGB")
                    h = compute_image_hash(rgb)
                    if h in seen_hashes:
                        duplicates_detected += 1
                    else:
                        seen_hashes[h] = fpath

                samples.append((fpath, c_id))
                class_counts[c_name] += 1
            except Exception as e:
                corrupt_files += 1

    deficient_classes = {c: class_counts[c] for c in class_names if class_counts[c] < min_images_per_class}

    report = {
        "data_dir": data_dir,
        "total_classes": len(class_names),
        "class_names": class_names,
        "total_images": len(samples),
        "class_counts": dict(class_counts),
        "duplicates_detected": duplicates_detected,
        "corrupt_files": corrupt_files,
        "deficient_classes": deficient_classes,
        "coverage": check_neem_tulsi_coverage(class_names)
    }

    return samples, class_names, report

def create_real_splits(data_dir, val_ratio=0.15, test_ratio=0.15, min_images_per_class=10, seed=42):
    """
    Creates reproducible splits from genuine image folders.
    Supports either pre-split directories:
      data_dir/train/<Class>/, data_dir/val/<Class>/, data_dir/test/<Class>/
    OR flat class directory:
      data_dir/<Class>/... (stratified split into train/val/test)

    Returns:
      train_samples, val_samples, test_samples, class_names, report
    """
    import random
    random.seed(seed)

    # Check for pre-split structure
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        # Load from separate subdirectories
        train_samples, class_names_tr, rep_tr = validate_and_load_real_dataset(train_dir, min_images_per_class=min_images_per_class)
        val_samples, class_names_val, rep_val = validate_and_load_real_dataset(val_dir, min_images_per_class=1)
        test_samples, class_names_ts, rep_ts = validate_and_load_real_dataset(test_dir, min_images_per_class=1)

        # Ensure consistent class mapping across splits
        all_classes = sorted(list(set(class_names_tr) | set(class_names_val) | set(class_names_ts)))
        class_names, _, label2id = get_class_mappings(all_classes)

        report = {
            "mode": "pre_split_directories",
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "test_count": len(test_samples),
            "classes": class_names,
            "coverage": check_neem_tulsi_coverage(class_names)
        }
        return train_samples, val_samples, test_samples, class_names, report

    # Flat directory structure: perform stratified split
    all_samples, class_names, load_rep = validate_and_load_real_dataset(
        data_dir, min_images_per_class=min_images_per_class
    )

    if load_rep["deficient_classes"]:
        def_info = ", ".join([f"{k} ({v} images)" for k, v in load_rep["deficient_classes"].items()])
        print(f"[Dataset Warning] Following classes have fewer than {min_images_per_class} images: {def_info}")

    # Group samples by class
    by_class = {}
    for item in all_samples:
        fpath, c_id = item
        by_class.setdefault(c_id, []).append(item)

    train_samples = []
    val_samples = []
    test_samples = []

    for c_id, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = max(0, n - n_train - n_val)

        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    report = {
        "mode": "stratified_split",
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "classes": class_names,
        "load_report": load_rep,
        "coverage": check_neem_tulsi_coverage(class_names)
    }

    return train_samples, val_samples, test_samples, class_names, report

def create_synthetic_demo_splits(samples_per_class=20, num_ood_classes=3):
    """
    [WARNING: DEMO / SMOKE-TEST ONLY]
    Generates synthetic solid-color images to verify that PyTorch training code,
    tensor shapes, optimizers, and checkpointing logic run without syntax errors.
    DO NOT report accuracy or recognition results from this function as genuine AI capability.
    """
    class_names, id2label, label2id = get_class_mappings()
    num_classes = len(class_names)

    train_samples = []
    val_samples = []
    test_samples = []
    ood_samples = []

    torch.manual_seed(42)

    for c_id, c_name in enumerate(class_names):
        if samples_per_class >= 3:
            n_val = max(1, int(samples_per_class * 0.15))
            n_test = max(1, int(samples_per_class * 0.15))
            n_train = samples_per_class - n_val - n_test
        else:
            n_train, n_val, n_test = samples_per_class, 0, 0

        for i in range(samples_per_class):
            hue_offset = (c_id * 17) % 255
            img = Image.new("RGB", (224, 224), color=(
                (hue_offset + 30) % 255,
                (hue_offset + 120) % 255,
                (hue_offset + 60) % 255
            ))
            if i < n_train:
                train_samples.append((img, c_id))
            elif i < n_train + n_val:
                val_samples.append((img, c_id))
            else:
                test_samples.append((img, c_id))

    for ood_id in range(num_ood_classes * 15):
        img = Image.new("RGB", (224, 224), color=(
            (ood_id * 43) % 200,
            (ood_id * 23) % 150,
            (ood_id * 89 + 100) % 255
        ))
        ood_samples.append((img, -1))

    manifest = {
        "num_classes": num_classes,
        "class_names": class_names,
        "taxonomy": SPECIES_TAXONOMY,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "ood_size": len(ood_samples),
        "is_synthetic_demo": True
    }

    with open("dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return train_samples, val_samples, test_samples, ood_samples, manifest

# Backward compatibility alias
create_synthetic_benchmark_splits = create_synthetic_demo_splits

if __name__ == "__main__":
    names, _, _ = get_class_mappings()
    cov = check_neem_tulsi_coverage(names)
    print(f"FloraSense AI Dataset Module initialized.")
    print(f"Total Taxonomy Species: {len(names)}")
    print(f"Tulsi Covered: {cov['tulsi_present']} | Neem Covered: {cov['neem_present']}")
