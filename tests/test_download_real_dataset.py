import io
import json
import tarfile

import pytest
from PIL import Image

from download_real_dataset import add_class_to_existing_splits


def make_image_bytes(color):
    image = Image.new("RGB", (64, 64), color)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def normalize_jpeg(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    output = io.BytesIO()
    image.save(output, "JPEG", quality=92)
    return output.getvalue()


def create_dataset_dirs(root):
    for split in ("train", "val", "test"):
        (root / split).mkdir()


def create_rattlesnake_archive(path, image_count):
    with tarfile.open(path, "w") as archive:
        for index in range(image_count):
            image_data = make_image_bytes((index * 30, index * 13, index * 7))
            image_info = tarfile.TarInfo(
                f"house_plant_species/Rattlesnake Plant (Calathea lancifolia)/{index}.jpg"
            )
            image_info.size = len(image_data)
            archive.addfile(image_info, io.BytesIO(image_data))


def test_add_class_preserves_existing_splits_and_updates_manifest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    create_dataset_dirs(data_dir)
    existing_file = data_dir / "train" / "Existing Plant"
    existing_file.mkdir()
    (existing_file / "keep.txt").write_text("preserve", encoding="utf-8")
    (existing_file / "duplicate.jpg").write_bytes(
        normalize_jpeg(make_image_bytes((0, 0, 0)))
    )
    manifest = {
        "splits": {"train": {"Existing Plant": 2}, "val": {}, "test": {}},
        "class_counts": {"Existing Plant": 2},
        "missing_classes": ["Rattlesnake Plant"],
        "insufficient_classes": [],
        "total_images": 2,
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    archive_path = tmp_path / "plants.tar"
    create_rattlesnake_archive(archive_path, 8)

    result = add_class_to_existing_splits(
        str(archive_path),
        str(data_dir),
        "Rattlesnake Plant",
        min_images=3,
    )

    assert result["image_count"] == 7
    assert result["splits"] == {"train": 5, "val": 1, "test": 1}
    assert (existing_file / "keep.txt").read_text(encoding="utf-8") == "preserve"
    assert all(
        len(list((data_dir / split / "Rattlesnake Plant").glob("*.jpg"))) == count
        for split, count in result["splits"].items()
    )
    updated_manifest = json.loads(
        (data_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert updated_manifest["class_counts"]["Rattlesnake Plant"] == 7
    assert updated_manifest["total_images"] == 9
    assert "Rattlesnake Plant" not in updated_manifest["missing_classes"]


def test_add_class_refuses_to_overwrite_existing_class(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    create_dataset_dirs(data_dir)
    rattlesnake_dir = data_dir / "train" / "Rattlesnake Plant"
    rattlesnake_dir.mkdir()
    archive_path = tmp_path / "plants.tar"
    create_rattlesnake_archive(archive_path, 4)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        add_class_to_existing_splits(
            str(archive_path),
            str(data_dir),
            "Rattlesnake Plant",
            min_images=3,
        )
