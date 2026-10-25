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
import random
from collections import Counter
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_evaluation_transform,
    create_training_transform,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Legacy taxonomy metadata; production class mappings are discovered from dataset folders.
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
    Otherwise defaults to the legacy SPECIES_TAXONOMY keys. Real training should
    pass the discovered dataset class names.
    """
    if class_names is None:
        names = sorted(list(SPECIES_TAXONOMY.keys()))
    else:
        names = sorted(list(set(class_names)))
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return names, id2label, label2id

def compute_image_hash(image: Image.Image) -> str:
    """Hash decoded RGB pixels and dimensions to identify exact image duplicates."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()

def get_data_transforms():
    """
    Constructs clean, botanically conservative transforms.
    Avoids extreme distortion that destroys leaf shape and venation patterns.
    """
    return create_training_transform(), create_evaluation_transform()

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

def _class_directories(data_dir):
    return sorted(
        entry.name for entry in os.scandir(data_dir)
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _print_dataset_report(report):
    print(
        f"[Dataset] {report['total_images']} valid unique images, "
        f"{report['total_classes']} classes, "
        f"{report['duplicates_detected']} duplicate files excluded, "
        f"{len(report['invalid_files'])} invalid images."
    )
    for class_name in report["class_names"]:
        counts = report["split_counts"].get(class_name)
        if counts:
            split_summary = ", ".join(f"{split}={count}" for split, count in counts.items())
            print(f"  {class_name}: {split_summary}, total={report['class_counts'][class_name]}")
        else:
            print(f"  {class_name}: {report['class_counts'][class_name]}")
    for item in report["invalid_files"]:
        print(f"  [Invalid] {item['path']}: {item['error']}")
    for item in report["conflicting_duplicates"]:
        print(f"  [Conflicting duplicate excluded] {item['paths']}")


def validate_and_load_real_dataset(
    data_dir,
    min_images_per_class=10,
    valid_extensions=(".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jpe", ".jfif"),
    class_names=None,
):
    """Validate class-folder images and exclude exact decoded-pixel duplicates.

    ``class_names`` fixes the label mapping when loading multiple splits. The
    returned report includes zero-count classes, duplicate and invalid files,
    and per-class counts.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    subdirs = _class_directories(data_dir)
    if not subdirs:
        raise ValueError(f"No class subdirectories found in '{data_dir}'. Expected format: '{data_dir}/<ClassName>/image.jpg'")

    if class_names is None:
        class_names = subdirs
    else:
        unexpected = sorted(set(subdirs) - set(class_names))
        if unexpected:
            raise ValueError(f"Unexpected class folders in '{data_dir}': {unexpected}")
    class_names, _, label2id = get_class_mappings(class_names)

    hash_groups = {}
    invalid_files = []
    duplicate_files = []
    conflicting_duplicates = []
    class_counts = Counter()
    skipped_extensions = 0

    for c_name in class_names:
        c_dir = os.path.join(data_dir, c_name)
        if not os.path.isdir(c_dir):
            continue

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
                with Image.open(fpath) as img:
                    img.load()
                    if img.width < 1 or img.height < 1:
                        raise ValueError("Image has an empty dimension.")
                    image_hash = compute_image_hash(img)
                candidate = (fpath, label2id[c_name], c_name)
                hash_groups.setdefault(image_hash, []).append(candidate)
            except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as error:
                invalid_files.append({"path": fpath, "error": str(error)})

    samples = []
    for group in hash_groups.values():
        if len({item[2] for item in group}) > 1:
            conflicting_duplicates.append({"paths": [item[0] for item in group]})
            duplicate_files.extend(item[0] for item in group[1:])
            continue
        samples.append(group[0][:2])
        duplicate_files.extend(item[0] for item in group[1:])
        class_counts[group[0][2]] += 1

    deficient_classes = {
        c: class_counts[c] for c in class_names
        if class_counts[c] < min_images_per_class
    }

    report = {
        "data_dir": data_dir,
        "total_classes": len(class_names),
        "class_names": class_names,
        "total_images": len(samples),
        "class_counts": {name: class_counts[name] for name in class_names},
        "duplicates_detected": len(duplicate_files),
        "duplicate_files": duplicate_files,
        "conflicting_duplicates": conflicting_duplicates,
        "invalid_files": invalid_files,
        "invalid_file_count": len(invalid_files),
        "skipped_extensions": skipped_extensions,
        "deficient_classes": deficient_classes,
        "coverage": check_neem_tulsi_coverage(class_names),
        "split_counts": {},
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
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    split_dirs = {"train": train_dir, "val": val_dir, "test": test_dir}
    existing_splits = [os.path.isdir(path) for path in split_dirs.values()]

    if any(existing_splits):
        if not all(existing_splits):
            raise ValueError(
                f"Expected all of train/val/test directories under '{data_dir}', "
                "or one flat directory of class folders."
            )

        all_classes = sorted(set().union(
            *(_class_directories(path) for path in split_dirs.values())
        ))
        if not all_classes:
            raise ValueError(f"No class folders found in the splits under '{data_dir}'.")
        loaded = {
            split: validate_and_load_real_dataset(
                path,
                min_images_per_class=0,
                class_names=all_classes,
            )
            for split, path in split_dirs.items()
        }
        split_samples = {
            split: loaded[split][0]
            for split in split_dirs
        }
        split_reports = {
            split: loaded[split][2]
            for split in split_dirs
        }

        # Keep the held-out test copy when an identical image occurs in multiple
        # partitions; remove duplicates with conflicting class labels entirely.
        duplicate_groups = {}
        for split, samples in split_samples.items():
            for path, class_id in samples:
                with Image.open(path) as image:
                    image_hash = compute_image_hash(image)
                duplicate_groups.setdefault(image_hash, []).append((split, path, class_id))
        priority = {"test": 0, "val": 1, "train": 2}
        keep_paths = {split: set() for split in split_dirs}
        cross_split_duplicates = 0
        cross_split_conflicts = []
        for group in duplicate_groups.values():
            classes = {item[2] for item in group}
            if len(classes) > 1:
                cross_split_conflicts.append([item[1] for item in group])
                continue
            selected = min(group, key=lambda item: (priority[item[0]], item[1]))
            keep_paths[selected[0]].add(selected[1])
            cross_split_duplicates += len(group) - 1

        for split in split_dirs:
            split_samples[split] = [
                (path, class_id)
                for path, class_id in split_samples[split]
                if path in keep_paths[split]
            ]

        _, _, class_to_id = get_class_mappings(all_classes)
        split_counts = {
            name: {
                split: sum(class_id == class_to_id[name] for _, class_id in split_samples[split])
                for split in split_dirs
            }
            for name in all_classes
        }
        deficient = {
            name: counts for name, counts in split_counts.items()
            if counts["train"] < min_images_per_class or counts["val"] < 1 or counts["test"] < 1
        }
        report = {
            "mode": "pre_split_directories",
            "classes": all_classes,
            "class_names": all_classes,
            "total_classes": len(all_classes),
            "train_count": len(split_samples["train"]),
            "val_count": len(split_samples["val"]),
            "test_count": len(split_samples["test"]),
            "total_images": sum(map(len, split_samples.values())),
            "class_counts": {
                name: sum(split_counts[name].values()) for name in all_classes
            },
            "split_counts": split_counts,
            "duplicates_detected": cross_split_duplicates + sum(
                part["duplicates_detected"] for part in split_reports.values()
            ),
            "invalid_files": [
                entry for part in split_reports.values() for entry in part["invalid_files"]
            ],
            "invalid_file_count": sum(part["invalid_file_count"] for part in split_reports.values()),
            "conflicting_duplicates": (
                [entry for part in split_reports.values() for entry in part["conflicting_duplicates"]]
                + [{"paths": paths} for paths in cross_split_conflicts]
            ),
            "deficient_classes": deficient,
            "coverage": check_neem_tulsi_coverage(all_classes),
        }
        _print_dataset_report(report)
        if deficient:
            details = "; ".join(
                f"{name} (train={counts['train']}, val={counts['val']}, test={counts['test']})"
                for name, counts in deficient.items()
            )
            raise ValueError(
                "Insufficient valid unique images to populate every requested class "
                f"across train/validation/test: {details}"
            )
        return (
            split_samples["train"],
            split_samples["val"],
            split_samples["test"],
            all_classes,
            report,
        )

    all_samples, class_names, load_rep = validate_and_load_real_dataset(
        data_dir, min_images_per_class=min_images_per_class
    )

    if load_rep["deficient_classes"]:
        details = ", ".join(f"{name} ({count} unique images)" for name, count in load_rep["deficient_classes"].items())
        raise ValueError(
            f"Classes below the minimum of {min_images_per_class} valid unique images: {details}"
        )
    if any(len([sample for sample in all_samples if sample[1] == class_id]) < 3 for class_id in range(len(class_names))):
        raise ValueError("Every class needs at least 3 valid unique images to create train/validation/test splits.")

    by_class = {class_id: [] for class_id in range(len(class_names))}
    for item in all_samples:
        by_class[item[1]].append(item)
    train_samples = []
    val_samples = []
    test_samples = []
    rng = random.Random(seed)

    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("Not enough samples to create non-empty train/validation/test splits.")

        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    split_counts = {
        class_name: {
            "train": sum(label == class_id for _, label in train_samples),
            "val": sum(label == class_id for _, label in val_samples),
            "test": sum(label == class_id for _, label in test_samples),
        }
        for class_id, class_name in enumerate(class_names)
    }
    report = {
        "mode": "stratified_split",
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "classes": class_names,
        "class_names": class_names,
        "total_classes": len(class_names),
        "total_images": len(all_samples),
        "class_counts": load_rep["class_counts"],
        "split_counts": split_counts,
        "duplicates_detected": load_rep["duplicates_detected"],
        "invalid_files": load_rep["invalid_files"],
        "invalid_file_count": load_rep["invalid_file_count"],
        "conflicting_duplicates": load_rep["conflicting_duplicates"],
        "deficient_classes": load_rep["deficient_classes"],
        "load_report": load_rep,
        "coverage": check_neem_tulsi_coverage(class_names)
    }

    _print_dataset_report(report)
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
