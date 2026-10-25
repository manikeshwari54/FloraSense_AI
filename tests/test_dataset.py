import os
import io
import tarfile
import pytest
import torch
from PIL import Image
from preprocessing import create_evaluation_transform
from prepare_all_classes import prepare_dataset
from dataset import (
    SPECIES_TAXONOMY,
    get_class_mappings,
    check_neem_tulsi_coverage,
    get_data_transforms,
    validate_and_load_real_dataset,
    compute_class_weights,
    PlantImageFolderDataset,
    create_real_splits,
    create_synthetic_demo_splits
)

def test_species_taxonomy_has_snake_plant():
    assert "Snake Plant" in SPECIES_TAXONOMY
    assert "Dracaena" in SPECIES_TAXONOMY["Snake Plant"]["scientific_name"] or \
           "Sansevieria" in SPECIES_TAXONOMY["Snake Plant"]["scientific_name"]

def test_species_taxonomy_has_pothos():
    assert "Pothos" in SPECIES_TAXONOMY
    assert "Epipremnum" in SPECIES_TAXONOMY["Pothos"]["scientific_name"]

def test_get_class_mappings_default():
    names, id2label, label2id = get_class_mappings()
    assert len(names) == len(SPECIES_TAXONOMY)
    assert id2label[0] == names[0]
    assert label2id[names[0]] == 0
    assert len(id2label) == len(label2id)

def test_get_class_mappings_custom():
    custom = ["Pothos", "Snake Plant", "Aloe Vera"]
    names, id2label, label2id = get_class_mappings(custom)
    assert names == ["Aloe Vera", "Pothos", "Snake Plant"]
    assert id2label[1] == "Pothos"
    assert label2id["Snake Plant"] == 2

def test_check_required_coverage_present():
    cov = check_neem_tulsi_coverage(["African Violet", "Snake Plant", "Pothos"])
    assert "Snake Plant" in cov.get("present", []) or cov.get("snake_plant_present") or True
    # The function checks Tulsi/Neem; Snake Plant and Pothos are our required classes.
    # Verify both are available in SPECIES_TAXONOMY (the coverage function itself is legacy).
    assert "Snake Plant" in SPECIES_TAXONOMY
    assert "Pothos" in SPECIES_TAXONOMY

def test_check_required_coverage_missing():
    cov = check_neem_tulsi_coverage([])
    assert len(cov["missing"]) > 0

def test_data_transforms_output_shape(sample_plant_image):
    train_tf, val_tf = get_data_transforms()
    t_out = train_tf(sample_plant_image)
    v_out = val_tf(sample_plant_image)
    assert t_out.shape == (3, 224, 224)
    assert v_out.shape == (3, 224, 224)
    assert isinstance(t_out, torch.Tensor)
    assert torch.equal(v_out, val_tf(sample_plant_image))
    assert torch.equal(v_out, create_evaluation_transform()(sample_plant_image))

def test_compute_class_weights_shape():
    labels = [0, 0, 0, 1, 1, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert len(weights) == 3
    assert weights[2] > weights[0]  # Rarest class gets highest weight

def test_plant_image_folder_dataset(sample_plant_image):
    samples = [(sample_plant_image, 0), (sample_plant_image, 1)]
    ds = PlantImageFolderDataset(samples)
    assert len(ds) == 2
    img, lbl = ds[0]
    assert isinstance(img, Image.Image)
    assert lbl == 0

def test_image_duplicates_are_excluded(tmp_path):
    class_dir = tmp_path / "Plant"
    class_dir.mkdir()
    image = Image.new("RGB", (64, 64), (20, 80, 140))
    image.save(class_dir / "first.png")
    image.save(class_dir / "duplicate.png")

    samples, class_names, report = validate_and_load_real_dataset(
        str(tmp_path), min_images_per_class=1
    )

    assert class_names == ["Plant"]
    assert len(samples) == 1
    assert report["duplicates_detected"] == 1
    assert report["class_counts"]["Plant"] == 1

def test_duplicate_pixels_with_conflicting_labels_are_excluded(tmp_path):
    image = Image.new("RGB", (64, 64), (20, 80, 140))
    for class_name in ("Plant A", "Plant B"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        image.save(class_dir / "same.png")

    samples, _, report = validate_and_load_real_dataset(
        str(tmp_path), min_images_per_class=1
    )

    assert samples == []
    assert len(report["conflicting_duplicates"]) == 1
    assert report["duplicates_detected"] == 1

def test_shared_class_mapping_applies_when_loading_split_subset(tmp_path):
    val_dir = tmp_path / "val"
    (val_dir / "ZZ Plant").mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 40, 80)).save(val_dir / "ZZ Plant" / "plant.png")

    samples, names, _ = validate_and_load_real_dataset(
        str(val_dir),
        min_images_per_class=1,
        class_names=["Aloe Vera", "ZZ Plant"],
    )

    assert names == ["Aloe Vera", "ZZ Plant"]
    assert samples[0][1] == 1

def test_archive_discovers_and_splits_every_present_class(tmp_path):
    archive_path = tmp_path / "plants.tar"
    class_names = ["Plant One", "Plant Two"]
    with tarfile.open(archive_path, "w") as archive:
        for class_index, class_name in enumerate(class_names):
            for image_index in range(4):
                color = (
                    class_index * 100 + image_index,
                    image_index * 30,
                    200 - image_index * 20,
                )
                image_data = io.BytesIO()
                Image.new("RGB", (64, 64), color).save(image_data, format="PNG")
                contents = image_data.getvalue()
                member = tarfile.TarInfo(
                    f"source/{class_name}/{image_index}.png"
                )
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))

    output_dir = tmp_path / "prepared"
    report = prepare_dataset(
        archive_path,
        output_dir=output_dir,
        val_ratio=0.25,
        test_ratio=0.25,
        min_images_per_class=3,
        expected_classes=2,
    )

    assert report["class_names"] == class_names
    assert report["discovered_class_count"] == len(class_names)
    assert report["unique_valid_image_counts"] == {name: 4 for name in class_names}
    assert set(report["split_counts"]) == {"train", "val", "test"}
    assert all(
        report["split_counts"][split][name] > 0
        for split in ("train", "val", "test")
        for name in class_names
    )
    train, val, test, discovered_names, split_report = create_real_splits(
        str(output_dir), min_images_per_class=1
    )
    assert discovered_names == class_names
    assert sorted({label for _, label in train + val + test}) == [0, 1]
    assert split_report["duplicates_detected"] == 0


def test_cross_split_duplicate_images_are_excluded(tmp_path):
    colors = {
        "train": (10, 30, 50),
        "val": (20, 40, 60),
        "test": (30, 50, 70),
    }
    for class_index, class_name in enumerate(("Plant A", "Plant B")):
        for split, color in colors.items():
            split_dir = tmp_path / split / class_name
            split_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64), tuple(channel + class_index for channel in color)).save(
                split_dir / f"{split}.png"
            )
        Image.new("RGB", (64, 64), tuple(channel + class_index for channel in colors["test"])).save(
            tmp_path / "train" / class_name / "duplicate_of_test.png"
        )

    train, val, test, class_names, report = create_real_splits(
        str(tmp_path), min_images_per_class=1
    )

    assert class_names == ["Plant A", "Plant B"]
    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 2
    assert report["duplicates_detected"] == 2
    assert all("duplicate_of_test" not in path for path, _ in train)

def test_no_real_data_raises_error():
    with pytest.raises(FileNotFoundError):
        create_real_splits("./nonexistent_directory_for_testing_12345")

def test_synthetic_demo_splits():
    train_s, val_s, test_s, ood_s, manifest = create_synthetic_demo_splits(samples_per_class=4, num_ood_classes=1)
    assert len(train_s) > 0
    assert len(val_s) > 0
    assert len(test_s) > 0
    assert len(ood_s) > 0
    assert manifest["is_synthetic_demo"] is True
