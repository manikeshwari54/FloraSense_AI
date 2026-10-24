"""
FloraSense AI — Indoor Plant Dataset Downloader
-----------------------------------------------
Dataset: kakasher/house-plant-species (CC BY 3.0)
  14,790 images, 47 classes, from Bing Images (manually curated, non-expert)

Selected 20 indoor classes (excludes outdoor/seasonal species).

Usage:
  python download_real_dataset.py [--data_dir ./data] [--min_images 50]

This script:
  1. Downloads house_plant_species.tar from Hugging Face
  2. Extracts to a temp directory
  3. Selects the 20 chosen indoor species
  4. Validates images (PIL open, minimum size check)
  5. Deduplicates via MD5 hash
  6. Reproducibly splits 70/15/15 stratified by class
  7. Saves to data/train/, data/val/, data/test/
  8. Reports per-class image counts and any deficiencies
  9. Saves dataset_manifest.json with per-class counts and split info

IMPORTANT: Never synthesizes replacement images.
"""

import os
import sys
import io
import json
import tarfile
import hashlib
import shutil
import random
import argparse
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from PIL import Image

# ---- Selected 32 indoor classes ----
# Maps folder name IN THE TAR to our canonical class name
SELECTED_CLASSES = {
    # Original 19 active classes
    "African Violet (Saintpaulia ionantha)": "African Violet",
    "Aloe Vera": "Aloe Vera",
    "Areca Palm (Dypsis lutescens)": "Areca Palm",
    "Boston Fern (Nephrolepis exaltata)": "Boston Fern",
    "Calathea": "Calathea",
    "Chinese evergreen (Aglaonema)": "Chinese Evergreen",
    "Dracaena": "Dracaena",
    "English Ivy (Hedera helix)": "English Ivy",
    "Jade plant (Crassula ovata)": "Jade Plant",
    "Monstera Deliciosa (Monstera deliciosa)": "Monstera Deliciosa",
    "Orchid": "Orchid",
    "Peace lily": "Peace Lily",
    "Pothos (Ivy arum)": "Pothos",
    "Prayer Plant (Maranta leuconeura)": "Prayer Plant",
    "Rubber Plant (Ficus elastica)": "Rubber Plant",
    "Snake plant (Sanseviera)": "Snake Plant",
    "Tradescantia": "Tradescantia",
    "Venus Flytrap": "Venus Flytrap",
    "ZZ Plant (Zamioculcas zamiifolia)": "ZZ Plant",
    # 13 Expansion Species (verified indoor houseplants)
    "Anthurium (Anthurium andraeanum)": "Anthurium",
    "Birds Nest Fern (Asplenium nidus)": "Birds Nest Fern",
    "Chinese Money Plant (Pilea peperomioides)": "Chinese Money Plant",
    "Christmas Cactus (Schlumbergera bridgesii)": "Christmas Cactus",
    "Dumb Cane (Dieffenbachia spp.)": "Dumb Cane",
    "Elephant Ear (Alocasia spp.)": "Elephant Ear",
    "Money Tree (Pachira aquatica)": "Money Tree",
    "Parlor Palm (Chamaedorea elegans)": "Parlor Palm",
    "Poinsettia (Euphorbia pulcherrima)": "Poinsettia",
    "Polka Dot Plant (Hypoestes phyllostachya)": "Polka Dot Plant",
    "Ponytail Palm (Beaucarnea recurvata)": "Ponytail Palm",
    "Rattlesnake Plant (Calathea lancifolia)": "Rattlesnake Plant",
    "Schefflera": "Schefflera"
}

# Exact mapping from raw TAR directory names to canonical class names
RAW_TAR_MAP = {
    "African Violet (Saintpaulia ionantha)": "African Violet",
    "Aloe Vera": "Aloe Vera",
    "Areca Palm (Dypsis lutescens)": "Areca Palm",
    "Boston Fern (Nephrolepis exaltata)": "Boston Fern",
    "Calathea": "Calathea",
    "Chinese evergreen (Aglaonema)": "Chinese Evergreen",
    "Dracaena": "Dracaena",
    "English Ivy (Hedera helix)": "English Ivy",
    "Jade plant (Crassula ovata)": "Jade Plant",
    "Monstera Deliciosa (Monstera deliciosa)": "Monstera Deliciosa",
    "Orchid": "Orchid",
    "Peace lily": "Peace Lily",
    "Pothos (Ivy arum)": "Pothos",
    "Prayer Plant (Maranta leuconeura)": "Prayer Plant",
    "Rubber Plant (Ficus elastica)": "Rubber Plant",
    "Snake plant (Sanseviera)": "Snake Plant",
    "Tradescantia": "Tradescantia",
    "Venus Flytrap": "Venus Flytrap",
    "ZZ Plant (Zamioculcas zamiifolia)": "ZZ Plant",
    "Anthurium (Anthurium andraeanum)": "Anthurium",
    "Birds Nest Fern (Asplenium nidus)": "Birds Nest Fern",
    "Chinese Money Plant (Pilea peperomioides)": "Chinese Money Plant",
    "Christmas Cactus (Schlumbergera bridgesii)": "Christmas Cactus",
    "Dumb Cane (Dieffenbachia spp.)": "Dumb Cane",
    "Elephant Ear (Alocasia spp.)": "Elephant Ear",
    "Money Tree (Pachira aquatica)": "Money Tree",
    "Parlor Palm (Chamaedorea elegans)": "Parlor Palm",
    "Poinsettia (Euphorbia pulcherrima)": "Poinsettia",
    "Polka Dot Plant (Hypoestes phyllostachya)": "Polka Dot Plant",
    "Ponytail Palm (Beaucarnea recurvata)": "Ponytail Palm",
    "Rattlesnake Plant (Calathea lancifolia)": "Rattlesnake Plant",
    "Schefflera": "Schefflera"
}

CLASS_FUZZY_MAP = {k.lower(): v for k, v in RAW_TAR_MAP.items()}
for k, v in list(RAW_TAR_MAP.items()):
    CLASS_FUZZY_MAP[v.lower()] = v
    CLASS_FUZZY_MAP[v.lower().replace(" ", "_")] = v

DATASET_URL = "https://huggingface.co/datasets/kakasher/house-plant-species/resolve/main/house_plant_species.tar"
MIN_IMAGE_DIM = 50


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def resolve_folder_name(folder: str) -> str:
    """Exact and fuzzy matching for TAR folders to canonical species names."""
    if folder in RAW_TAR_MAP:
        return RAW_TAR_MAP[folder]
    f_clean = folder.strip().lower()
    if f_clean in CLASS_FUZZY_MAP:
        return CLASS_FUZZY_MAP[f_clean]
    # Check exact canonical name match before generic substring match
    for raw, canon in RAW_TAR_MAP.items():
        if f_clean == canon.lower() or f_clean == raw.lower():
            return canon
    for raw, canon in RAW_TAR_MAP.items():
        if canon.lower() in f_clean:
            return canon
    return None


def validate_image(data: bytes) -> tuple:
    """Returns (is_valid, pil_image_or_none, error_msg)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return False, None, f"Too small: {w}x{h}"
        return True, img, None
    except Exception as e:
        return False, None, str(e)


def download_tar(url: str, dest: str):
    print(f"Downloading dataset from: {url}")
    print("(~150 MB — may take a few minutes on slow connections)")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 FloraSense/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded//1024//1024} MB / {total//1024//1024} MB ({pct:.1f}%)", end="", flush=True)
    print()


def extract_selected_classes(tar_path: str, extract_dir: str) -> dict:
    """
    Extracts images for selected classes from tar.
    Returns {canonical_class: [image_bytes_list]}
    Also prints all folder names found in tar for diagnostics.
    """
    class_images = defaultdict(list)
    found_folders = set()
    skipped_folders = set()

    print("\nInspecting TAR contents...")
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        print(f"Total entries in tar: {len(members)}")

        # First pass: discover folder structure
        for m in members:
            parts = m.name.replace("\\", "/").split("/")
            # Skip hidden files and non-image files
            if m.isfile():
                name = m.name.lower()
                if not any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
                    continue
                # Folder is parent directory of file
                folder = parts[-2] if len(parts) >= 2 else ""
                canonical = resolve_folder_name(folder)
                if canonical:
                    found_folders.add(folder)
                    try:
                        f = tf.extractfile(m)
                        if f:
                            data = f.read()
                            class_images[canonical].append(data)
                    except Exception:
                        pass
                else:
                    skipped_folders.add(folder)

    print(f"\nFolders matched to selected classes: {sorted(found_folders)}")
    unmatched = skipped_folders - found_folders
    if unmatched:
        print(f"Unmatched/excluded folders (first 20): {sorted(unmatched)[:20]}")
    return dict(class_images)


def split_and_save(class_images: dict, output_root: str, val_ratio: float, test_ratio: float, seed: int, min_images: int):
    random.seed(seed)

    CANONICAL_CLASSES = sorted(set(CLASS_FUZZY_MAP.values()))

    manifest = {
        "dataset": "kakasher/house-plant-species",
        "license": "CC BY 3.0 — Kacper Gregorowicz",
        "source_url": "https://huggingface.co/datasets/kakasher/house-plant-species",
        "warning": "Images sourced from Bing Images, curated by non-expert. Verify label quality before production use.",
        "splits": {"train": {}, "val": {}, "test": {}},
        "class_counts": {},
        "missing_classes": [],
        "insufficient_classes": [],
        "total_images": 0
    }

    print("\n" + "="*60)
    print("Per-class image counts (before split):")
    print("="*60)

    seen_hashes = set()
    all_classes_data = {}

    for cls_name in CANONICAL_CLASSES:
        raw_images = class_images.get(cls_name, [])
        manifest["class_counts"][cls_name] = len(raw_images)

        if len(raw_images) == 0:
            print(f"  {cls_name:35s}: NOT FOUND IN DATASET")
            manifest["missing_classes"].append(cls_name)
            continue

        # Validate and deduplicate
        valid = []
        dupes = 0
        corrupt = 0
        for data in raw_images:
            h = compute_md5(data)
            if h in seen_hashes:
                dupes += 1
                continue
            ok, img, err = validate_image(data)
            if not ok:
                corrupt += 1
                continue
            seen_hashes.add(h)
            valid.append((data, img))

        note = ""
        if dupes > 0:
            note += f" ({dupes} dupes removed)"
        if corrupt > 0:
            note += f" ({corrupt} corrupt skipped)"

        n = len(valid)
        print(f"  {cls_name:35s}: {n:4d} valid images{note}")
        manifest["class_counts"][cls_name] = n

        if n < min_images:
            print(f"  WARNING: {cls_name} has only {n} images (minimum {min_images} recommended)")
            manifest["insufficient_classes"].append({"class": cls_name, "count": n, "minimum": min_images})

        all_classes_data[cls_name] = valid

    print("="*60)

    # Save splits
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)

    total_saved = 0

    for cls_name, valid_images in all_classes_data.items():
        n = len(valid_images)
        indices = list(range(n))
        random.shuffle(indices)

        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            print(f"  SKIP {cls_name}: too few images after splitting ({n} total)")
            continue

        split_indices = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:]
        }

        for split, idxs in split_indices.items():
            cls_dir = os.path.join(output_root, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for i, idx in enumerate(idxs):
                data, img = valid_images[idx]
                fname = f"{cls_name.replace(' ', '_')}_{i:04d}.jpg"
                fpath = os.path.join(cls_dir, fname)
                try:
                    img.save(fpath, "JPEG", quality=92)
                    total_saved += 1
                except Exception as e:
                    print(f"  Save error {fname}: {e}")
            manifest["splits"][split][cls_name] = len(idxs)

    manifest["total_images"] = total_saved

    # Save manifest
    manifest_path = os.path.join(output_root, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Copy manifest to project root too
    try:
        shutil.copy(manifest_path, "dataset_manifest.json")
    except Exception:
        pass

    return manifest


def add_class_to_existing_splits(
    tar_path: str,
    output_root: str,
    class_name: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_images: int = 10,
):
    """Add one class from the source archive without rewriting existing classes."""
    if class_name not in set(RAW_TAR_MAP.values()):
        raise ValueError(f"Unsupported class name: {class_name}")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")

    split_dirs = {
        split: os.path.join(output_root, split)
        for split in ("train", "val", "test")
    }
    if not all(os.path.isdir(path) for path in split_dirs.values()):
        raise FileNotFoundError(
            f"Expected existing train, val, and test directories under '{output_root}'."
        )

    class_dirs = {
        split: os.path.join(path, class_name)
        for split, path in split_dirs.items()
    }
    if any(os.path.exists(path) for path in class_dirs.values()):
        raise FileExistsError(
            f"'{class_name}' already exists in one or more dataset splits; "
            "refusing to overwrite existing images."
        )

    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    existing_hashes = set()
    for split_dir in split_dirs.values():
        for existing_path in Path(split_dir).rglob("*"):
            if existing_path.is_file() and existing_path.suffix.lower() in image_extensions:
                digest = hashlib.sha256(existing_path.read_bytes()).hexdigest()
                existing_hashes.add(digest)

    valid_images = []
    seen_hashes = set()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(image_extensions):
                continue
            parts = member.name.replace("\\", "/").split("/")
            folder = parts[-2] if len(parts) >= 2 else ""
            if resolve_folder_name(folder) != class_name:
                continue
            image_file = archive.extractfile(member)
            if image_file is None:
                continue
            image_data = image_file.read()
            valid, image, _ = validate_image(image_data)
            if not valid:
                continue
            output = io.BytesIO()
            image.save(output, "JPEG", quality=92)
            normalized_image = output.getvalue()
            image_hash = hashlib.sha256(normalized_image).hexdigest()
            if image_hash in existing_hashes or image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)
            valid_images.append(normalized_image)

    if len(valid_images) < min_images:
        raise ValueError(
            f"Found only {len(valid_images)} valid, unique images for '{class_name}'; "
            f"at least {min_images} are required."
        )

    indices = list(range(len(valid_images)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, int(len(indices) * test_ratio))
    val_count = max(1, int(len(indices) * val_ratio))
    train_count = len(indices) - test_count - val_count
    if train_count < 1:
        raise ValueError(f"Not enough images to create three splits for '{class_name}'.")

    split_indices = {
        "train": indices[:train_count],
        "val": indices[train_count:train_count + val_count],
        "test": indices[train_count + val_count:],
    }
    split_counts = {split: len(items) for split, items in split_indices.items()}

    manifest_path = os.path.join(output_root, "dataset_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    manifest.setdefault("splits", {})
    manifest.setdefault("class_counts", {})
    manifest.setdefault("missing_classes", [])
    manifest.setdefault("insufficient_classes", [])

    with tempfile.TemporaryDirectory(dir=output_root) as staging_dir:
        for split, image_indices in split_indices.items():
            staged_class_dir = os.path.join(staging_dir, split, class_name)
            os.makedirs(staged_class_dir, exist_ok=True)
            for image_number, image_index in enumerate(image_indices):
                filename = f"{class_name.replace(' ', '_')}_{image_number:04d}.jpg"
                with open(os.path.join(staged_class_dir, filename), "wb") as image_file:
                    image_file.write(valid_images[image_index])

        for split, destination in class_dirs.items():
            os.replace(os.path.join(staging_dir, split, class_name), destination)
            manifest["splits"].setdefault(split, {})[class_name] = split_counts[split]

        manifest["class_counts"][class_name] = len(valid_images)
        manifest["missing_classes"] = [
            missing for missing in manifest["missing_classes"] if missing != class_name
        ]
        manifest["insufficient_classes"] = [
            item for item in manifest["insufficient_classes"]
            if item.get("class") != class_name
        ]
        manifest["total_images"] = sum(
            sum(classes.values()) for classes in manifest["splits"].values()
        )
        staged_manifest_path = os.path.join(staging_dir, "dataset_manifest.json")
        with open(staged_manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        os.replace(staged_manifest_path, manifest_path)

    return {"class_name": class_name, "image_count": len(valid_images), "splits": split_counts}


def main():
    parser = argparse.ArgumentParser(description="FloraSense AI Indoor Plant Dataset Downloader")
    parser.add_argument("--data_dir", default="./data", help="Output directory for dataset splits")
    parser.add_argument("--min_images", type=int, default=50, help="Minimum images per class (warn if fewer)")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_download", action="store_true", help="Skip download if tar already present")
    parser.add_argument("--tar_path", default="", help="Path to already-downloaded tar file")
    parser.add_argument(
        "--add_class",
        default="",
        help="Add just this class to existing train/val/test splits without replacing other classes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FloraSense AI — Indoor Plant Dataset Preparation")
    print("Dataset: kakasher/house-plant-species (CC BY 3.0)")
    print("=" * 60)
    if args.add_class:
        print(f"Adding one class: {args.add_class}")
    else:
        print(f"Selecting {len(set(CLASS_FUZZY_MAP.values()))} indoor plant classes.")
    print(f"Output: {args.data_dir}")
    print()

    if args.tar_path and os.path.exists(args.tar_path):
        tar_path = args.tar_path
        print(f"Using existing tar: {tar_path}")
    else:
        tar_path = "house_plant_species.tar"
        if args.skip_download and os.path.exists(tar_path):
            print(f"Skipping download, using existing: {tar_path}")
        else:
            try:
                download_tar(DATASET_URL, tar_path)
            except Exception as e:
                print(f"\nERROR: Failed to download dataset: {e}")
                print("\nAlternative: Download manually from:")
                print("  https://huggingface.co/datasets/kakasher/house-plant-species")
                print(f"Then run: python download_real_dataset.py --tar_path <path_to_tar> --skip_download")
                sys.exit(1)

    if not os.path.exists(tar_path):
        print(f"ERROR: TAR file not found: {tar_path}")
        sys.exit(1)

    if args.add_class:
        try:
            result = add_class_to_existing_splits(
                tar_path,
                args.data_dir,
                args.add_class,
                args.val_ratio,
                args.test_ratio,
                args.seed,
                args.min_images,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as error:
            print(f"\nERROR: Could not add class: {error}")
            sys.exit(1)
        print(
            f"\nAdded {result['image_count']} images for {result['class_name']}: "
            f"{result['splits']}"
        )
        print(f"Updated manifest: {args.data_dir}/dataset_manifest.json")
        print("Retrain each model architecture before using the new class for identification.")
        return

    class_images = extract_selected_classes(tar_path, args.data_dir)

    if not class_images:
        print("\nERROR: No images found matching selected indoor classes in the TAR.")
        print("The TAR folder structure may differ from expected. Check unmatched folders above.")
        sys.exit(1)

    manifest = split_and_save(
        class_images, args.data_dir,
        args.val_ratio, args.test_ratio, args.seed, args.min_images
    )

    print(f"\nDataset preparation complete!")
    print(f"Total images saved: {manifest['total_images']}")
    print(f"Missing classes: {manifest['missing_classes']}")
    if manifest['insufficient_classes']:
        print(f"Under-represented classes:")
        for c in manifest['insufficient_classes']:
            print(f"  {c['class']}: {c['count']} images")
    print(f"\nManifest saved to: {args.data_dir}/dataset_manifest.json")
    print(f"\nNext step: train the model:")
    print(f"  python train_models.py --data_dir {args.data_dir} --arch efficientnet_b0")


if __name__ == "__main__":
    main()
