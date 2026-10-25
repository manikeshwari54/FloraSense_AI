"""Validate and split every class found in the House Plant Species TAR archive."""

import argparse
import io
import json
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from dataset import compute_image_hash

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".jpe", ".jfif"}
MIN_IMAGE_DIMENSION = 50


def _write_report(path, report):
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


def prepare_dataset(
    archive_path,
    output_dir="./data_47",
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    min_images_per_class=10,
    expected_classes=47,
):
    """Validate images in an archive, deduplicate, and create stratified splits.

    Source folder names are retained as model labels. The archive is read only;
    output is written to a new directory and existing non-empty output is refused.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")
    if min_images_per_class < 3:
        raise ValueError("At least 3 images per class are needed for train/validation/test.")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. Choose a new directory; "
            "the existing contents will not be overwritten."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".preparing"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir()

    class_counts = Counter()
    raw_counts = Counter()
    invalid_files = []
    duplicate_files = []
    conflicting_groups = []
    staged_by_class = defaultdict(list)
    digest_records = {}
    conflict_digests = set()
    skipped_files = 0
    member_count = 0

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_count += 1
                parts = member.name.replace("\\", "/").strip("/").split("/")
                if len(parts) != 3 or Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped_files += 1
                    continue

                class_name = parts[1]
                if class_name in ("", ".", ".."):
                    skipped_files += 1
                    continue
                raw_counts[class_name] += 1
                source_file = archive.extractfile(member)
                if source_file is None:
                    invalid_files.append({"path": member.name, "error": "Archive member could not be read."})
                    continue

                try:
                    image_bytes = source_file.read()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        rgb = image.convert("RGB")
                        rgb.load()
                        if rgb.width < MIN_IMAGE_DIMENSION or rgb.height < MIN_IMAGE_DIMENSION:
                            raise ValueError(
                                f"Image is smaller than {MIN_IMAGE_DIMENSION} pixels: "
                                f"{rgb.width}x{rgb.height}"
                            )
                        image_hash = compute_image_hash(rgb)
                except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as error:
                    invalid_files.append({"path": member.name, "error": str(error)})
                    continue

                if image_hash in conflict_digests:
                    duplicate_files.append(member.name)
                    continue

                previous = digest_records.get(image_hash)
                if previous is not None:
                    duplicate_files.append(member.name)
                    if previous["class_name"] != class_name:
                        conflict_digests.add(image_hash)
                        conflicting_groups.append(
                            [previous["source"], member.name]
                        )
                        previous["staged_path"].unlink(missing_ok=True)
                        staged_by_class[previous["class_name"]].remove(previous["staged_path"])
                        class_counts[previous["class_name"]] -= 1
                        digest_records.pop(image_hash)
                    continue

                class_dir = staging_dir / class_name
                class_dir.mkdir(exist_ok=True)
                staged_path = class_dir / f"{len(staged_by_class[class_name]):06d}.jpg"
                rgb.save(staged_path, format="JPEG", quality=95)
                staged_by_class[class_name].append(staged_path)
                class_counts[class_name] += 1
                digest_records[image_hash] = {
                    "class_name": class_name,
                    "source": member.name,
                    "staged_path": staged_path,
                }

        class_names = sorted(raw_counts)
        report = {
            "dataset": "kakasher/house-plant-species",
            "source_archive": str(archive_path.resolve()),
            "source_url": "https://huggingface.co/datasets/kakasher/house-plant-species",
            "source_terms_note": (
                "The dataset card states that the data is for personal use only due to "
                "copyright considerations. Do not redistribute source images or derived "
                "artifacts without resolving rights."
            ),
            "archive_file_entries": member_count,
            "discovered_class_count": len(class_names),
            "expected_class_count": expected_classes,
            "class_names": class_names,
            "raw_image_counts": {name: raw_counts[name] for name in class_names},
            "unique_valid_image_counts": {name: class_counts[name] for name in class_names},
            "duplicate_file_count": len(duplicate_files),
            "duplicate_files": duplicate_files,
            "conflicting_duplicate_groups": conflicting_groups,
            "invalid_file_count": len(invalid_files),
            "invalid_files": invalid_files,
            "unsupported_or_non_image_entries": skipped_files,
            "minimum_images_per_class": min_images_per_class,
        }
        _write_report(output_dir / "dataset_audit_report.json", report)
        print(
            f"[Archive audit] {len(class_names)} discovered classes; "
            f"{sum(raw_counts.values())} candidate images; "
            f"{sum(class_counts.values())} valid unique images; "
            f"{len(duplicate_files)} duplicate files excluded; "
            f"{len(invalid_files)} invalid images."
        )
        for class_name in class_names:
            print(
                f"  {class_name}: raw={raw_counts[class_name]}, "
                f"unique_valid={class_counts[class_name]}"
            )
        for item in invalid_files:
            print(f"  [Invalid] {item['path']}: {item['error']}")
        for paths in conflicting_groups:
            print(f"  [Conflicting duplicate excluded] {paths}")

        insufficient = {
            name: class_counts[name]
            for name in class_names
            if class_counts[name] < min_images_per_class
        }
        if len(class_names) != expected_classes or insufficient:
            report["insufficient_classes"] = insufficient
            _write_report(output_dir / "dataset_audit_report.json", report)
            raise ValueError(
                "The archive does not provide the expected number of classes and "
                "sufficient valid unique images for every class. "
                f"Discovered {len(class_names)} (expected {expected_classes}); "
                f"insufficient classes: {insufficient}"
            )

        rng = random.Random(seed)
        split_counts = {split: {} for split in ("train", "val", "test")}
        for class_name in class_names:
            images = staged_by_class[class_name]
            rng.shuffle(images)
            count = len(images)
            test_count = max(1, int(count * test_ratio))
            val_count = max(1, int(count * val_ratio))
            train_count = count - test_count - val_count
            if train_count < 1:
                raise ValueError(
                    f"Not enough unique images to split '{class_name}' into three partitions."
                )
            indices = {
                "train": images[:train_count],
                "val": images[train_count:train_count + val_count],
                "test": images[train_count + val_count:],
            }
            for split, paths in indices.items():
                destination = output_dir / split / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for image_index, source_path in enumerate(paths):
                    shutil.move(
                        str(source_path),
                        str(destination / f"{class_name}_{image_index:05d}.jpg"),
                    )
                split_counts[split][class_name] = len(paths)

        report["split_counts"] = split_counts
        report["total_images"] = sum(class_counts.values())
        report["split_seed"] = seed
        report["validation_ratio"] = val_ratio
        report["test_ratio"] = test_ratio
        report["invalid_files"] = invalid_files
        report["duplicates_excluded"] = duplicate_files
        _write_report(output_dir / "dataset_manifest.json", report)
        print("[Dataset split] " + ", ".join(
            f"{split}={sum(counts.values())}" for split, counts in split_counts.items()
        ))
        return report
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Validate, deduplicate, and stratify all classes in the source TAR."
    )
    parser.add_argument("--archive", default="./house_plant_species.tar")
    parser.add_argument("--output_dir", default="./data_47")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_images_per_class", type=int, default=10)
    parser.add_argument("--expected_classes", type=int, default=47)
    args = parser.parse_args()
    prepare_dataset(
        archive_path=args.archive,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_images_per_class=args.min_images_per_class,
        expected_classes=args.expected_classes,
    )


if __name__ == "__main__":
    main()
