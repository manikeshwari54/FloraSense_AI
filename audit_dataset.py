"""Read-only audit of an existing train/validation/test image tree."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_DIMENSION = 50
LOW_RESOLUTION_DIMENSION = 224
NEAR_DUPLICATE_DISTANCE = 3


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image):
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def chunk_keys(value):
    return tuple((value >> offset) & 0xFFFF for offset in (0, 16, 32, 48))


def audit_dataset(data_dir):
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records = []
    per_class = defaultdict(lambda: {split: Counter() for split in SPLITS})
    exact_hashes = defaultdict(list)
    phash_records = defaultdict(list)
    image_sizes = defaultdict(list)

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for image_path in sorted(split_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(root).as_posix()
            parts = Path(relative_path).parts
            class_name = parts[1] if len(parts) > 2 else "(unclassified)"
            counts = per_class[class_name][split]
            counts["files"] += 1
            record = {"path": relative_path, "class": class_name, "split": split}

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
                    rgb = image.convert("RGB")
                    entropy = round(rgb.convert("L").entropy(), 3)
                    image_hash = average_hash(rgb)

                record.update({"width": width, "height": height, "entropy": entropy})
                counts["valid"] += 1
                image_sizes[class_name].append((width, height))
                flags = []
                if min(width, height) < MIN_DIMENSION:
                    flags.append("below_minimum_dimension")
                    counts["below_minimum_dimension"] += 1
                if min(width, height) < LOW_RESOLUTION_DIMENSION:
                    flags.append("below_224px_short_side")
                    counts["below_224px_short_side"] += 1
                if max(width, height) / min(width, height) > 3:
                    flags.append("extreme_aspect_ratio")
                    counts["extreme_aspect_ratio"] += 1
                if entropy < 2.0:
                    flags.append("very_low_grayscale_entropy")
                    counts["very_low_grayscale_entropy"] += 1
                if flags:
                    record["quality_flags"] = flags

                record["sha256"] = file_sha256(image_path)
                exact_hashes[record["sha256"]].append(record)
                phash_records[image_hash].append(record)
                records.append(record)
            except Exception as error:
                counts["corrupt"] += 1
                record["error"] = str(error)
                records.append(record)

    exact_groups = [group for group in exact_hashes.values() if len(group) > 1]
    exact_pairs = []
    exact_cross_split_pairs = 0
    for group in exact_groups:
        splits = {record["split"] for record in group}
        if len(splits) > 1:
            exact_cross_split_pairs += len(group) - 1
        exact_pairs.append([record["path"] for record in group])

    unique_hashes = sorted(phash_records)
    hash_buckets = defaultdict(set)
    for value in unique_hashes:
        for chunk_index, key in enumerate(chunk_keys(value)):
            hash_buckets[(chunk_index, key)].add(value)

    near_groups = []
    near_cross_split_pairs = 0
    seen_pairs = set()
    for value in unique_hashes:
        candidates = set()
        for chunk_index, key in enumerate(chunk_keys(value)):
            candidates.update(hash_buckets[(chunk_index, key)])
        for candidate in candidates:
            if candidate <= value:
                continue
            distance = (value ^ candidate).bit_count()
            if distance > NEAR_DUPLICATE_DISTANCE:
                continue
            seen_pairs.add((value, candidate))
            left = phash_records[value]
            right = phash_records[candidate]
            if any(a["split"] != b["split"] for a in left for b in right):
                near_cross_split_pairs += 1
            if len(near_groups) < 500:
                near_groups.append({
                    "hamming_distance": distance,
                    "images": [item["path"] for item in left[:3] + right[:3]],
                })

    classes = sorted(per_class)
    totals = {}
    for class_name in classes:
        split_counts = per_class[class_name]
        totals[class_name] = {
            split: dict(split_counts[split]) for split in SPLITS
        }
        totals[class_name]["total_valid"] = sum(
            split_counts[split]["valid"] for split in SPLITS
        )

    valid_class_counts = [values["total_valid"] for values in totals.values()]
    min_count = min(valid_class_counts, default=0)
    max_count = max(valid_class_counts, default=0)
    quality_warnings = sum(record.get("quality_flags") is not None for record in records)

    return {
        "dataset_root": str(root),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_note": "Image quality flags are screening heuristics; labels and suitability require human review.",
        "summary": {
            "class_count": len(classes),
            "valid_image_count": sum(valid_class_counts),
            "corrupt_file_count": sum(
                values[split]["corrupt"] for values in per_class.values() for split in SPLITS
            ),
            "exact_duplicate_groups": len(exact_groups),
            "exact_cross_split_duplicate_links": exact_cross_split_pairs,
            "near_duplicate_pairs_at_or_below_hamming_distance_3": len(seen_pairs),
            "near_duplicate_cross_split_pairs": near_cross_split_pairs,
            "images_with_quality_warnings": quality_warnings,
            "min_class_images": min_count,
            "max_class_images": max_count,
            "max_to_min_class_ratio": round(max_count / min_count, 3) if min_count else None,
        },
        "per_class": totals,
        "image_sizes": {
            class_name: {
                "minimum_width": min((size[0] for size in sizes), default=None),
                "minimum_height": min((size[1] for size in sizes), default=None),
                "below_224px_short_side": sum(min(size) < LOW_RESOLUTION_DIMENSION for size in sizes),
            }
            for class_name, sizes in sorted(image_sizes.items())
        },
        "exact_duplicate_groups": exact_pairs,
        "near_duplicate_examples_capped_at_500": near_groups,
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit an existing image dataset without modifying it.")
    parser.add_argument("--data_dir", default="./data", help="Root containing train/val/test folders")
    parser.add_argument("--output", default="dataset_audit_report.json", help="JSON report path")
    args = parser.parse_args()

    report = audit_dataset(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Per-class counts and file-level findings: {output_path}")


if __name__ == "__main__":
    main()