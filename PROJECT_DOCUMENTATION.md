# FloraSense AI: Project Documentation

**Document scope:** This document describes the root project in this workspace as inspected on 2026-10-18. It distinguishes current implementation from README claims, metadata, optional scripts, and unverified assumptions. It does not imply that a new training or evaluation run was performed.

## 1. Project Overview

### Purpose

FloraSense AI is a local plant-photo classification and care-reference application. A user selects a photo, the application screens it for basic image quality, and a trained image classifier returns ranked plant candidates. When the result is considered sufficiently certain, the app can show a care profile from the local `plant_data.py` database.

### Problem and motivation

Indoor plant identification can be difficult for people who do not know botanical terminology. Visual search alone can also return inconsistent results, and care guidance is spread across different sources. This project combines a constrained image classifier with a local species-to-care lookup in one demonstration application.

### Objectives

- Provide a browser-based interface for identifying a limited set of indoor plant labels.
- Preserve model class indices by loading the checkpoint together with its class mapping.
- Screen photos for minimum size, exposure, and blur before classification.
- Expose an uncertainty path instead of presenting every top-ranked class as certain.
- Present locally stored plant-care information after identification or in a separate care browser.
- Provide training, evaluation, API, and test code for experimentation and review.

### Intended users and scope

The interface is suitable for students, project reviewers, and indoor-plant owners. The actual classifier is limited to the 19 labels in the active EfficientNet-B0 mapping(is the list that connects the model’s output numbers to plant names. For example, if the model outputs class number 0, the mapping tells the app which plant name to display.). It is not a general botanical identifier, plant-disease diagnostic system, or veterinary/medical resource.

### Current UI features

The root Streamlit app has Home, Identify Plant, Plant Care, Supported Plants, Model Information, and About Project sections. The identification view accepts JPG/JPEG, PNG, and WebP files, shows a preview, returns a primary candidate and alternatives, retains uncertainty handling, and can display matching care notes. The application also offers MobileNetV3-Small as an optional checkpoint selection.

### Active model labels

The checkpoint mapping contains these 19 labels:

African Violet, Aloe Vera, Areca Palm, Boston Fern, Calathea, Chinese Evergreen, Dracaena, English Ivy, Jade Plant, Monstera Deliciosa, Orchid, Peace Lily, Pothos, Prayer Plant, Rubber Plant, Snake Plant, Tradescantia, Venus Flytrap, and ZZ Plant.

The README and care database also describe Rattlesnake Plant. The source archive's 315 images for this label are exact duplicates of images already labeled Calathea, so they were not retained as a separate training class. The active checkpoint mapping and UI still recognize 19 labels.

## 2. Problem Statement and Proposed Solution

Manual identification requires familiarity with leaf shape, venation, growth habit, flowers, and botanical naming. Many houseplants share similar foliage, and a photo can be blurry, poorly exposed, cropped, or show an unusual growth stage. Care needs also vary by species and by indoor conditions.

FloraSense provides a limited, learned image classifier and a local care database. The classifier produces a ranked list rather than a textual explanation of plant biology. A separate quality and confidence/OOD path can mark inputs unsuitable or uncertain. The care database supplies reference guidance only; the software does not examine symptoms, soil, watering history, or environmental sensor readings.

### Assumptions and limitations

- The photo depicts one plant and has enough visible foliage for the model to use.
- The correct plant, if supported, is one of the checkpoint's 19 class labels.
- Dataset labels are useful but not guaranteed botanically correct.
- A high model score is not a guarantee of correctness.
- Similar-looking foliage can be confused. In the saved report, Pothos is a particularly weak class.
- Care and toxicity notes are static text. The application does not verify them against current external sources at runtime.
- The local dataset has possible near-duplicate split leakage; see [Dataset documentation](#8-dataset-documentation).

## 3. Technology Stack

| Technology | Repository use | Notes |
|---|---|---|
| Python | Application, model, data, API, and test code | The configured workspace virtual environment is Python 3.13.9. The repository does not declare a `requires-python` range. |
| Streamlit | Main user interface in `app.py` | Sidebar navigation, image upload, state, widgets, tabs, and cached model loading. |
| CSS | `styles.css` | Local botanical colors and responsive presentation; no new font, image, or UI dependency is downloaded. |
| PyTorch | Checkpoint loading, tensor operations, model inference, and training | Training uses CUDA if available unless `--force_cpu` is supplied. Inference selects CUDA if available, otherwise CPU. |
| Torchvision | EfficientNet-B0 and MobileNetV3-Small builders and image transforms | Training requests the library's default pretrained weights; inference builds the architecture without downloading pretrained weights and then loads the local checkpoint. |
| Pillow | Image decoding, verification, conversion, and normalization inputs | The Streamlit app converts uploads to RGB before screening/inference. |
| NumPy | Image-quality calculations and evaluation aggregation | Used in the OOD image-quality and metric code. |
| scikit-learn | Classification report and confusion matrix | Used by `evaluate_models.py`; macro and weighted averages are reported. |
| Matplotlib / Seaborn | Evaluation plots | Confusion matrices and training/validation curves. |
| FastAPI / Uvicorn / Pydantic | Optional REST service in `backend/api.py` | Separate from the Streamlit interface. |
| requests / Pillow | GBIF candidate image collector | `collect_gbif_images.py` requests occurrence records and licensed media. |
| Transformers / timm / datasets / safetensors | Optional ViT and Hugging Face paths | These packages are listed in `requirements.txt`; they are not required by the root Streamlit CNN inference path itself. |
| pytest | Unit tests in `tests/` | The most recent recorded run after the UI update was 35 passing tests. |

### Observed local environment

The configured `.venv` was Python 3.13.9. At inspection, its package versions included Streamlit 1.66.0, PyTorch 2.14.1, Torchvision 0.29.1, Pillow 12.3.0, scikit-learn 1.9.1, Transformers 5.20.1, and pytest 9.1.1. `requirements.txt` sets version ranges, not an exact lockfile; another installation may resolve different compatible versions.

## 4. Project Structure

The following tree describes the active project root and omits individual image files, virtual-environment internals, and cache contents. The separate nested `FloraSense-AI/` copy was removed on 2026-10-24; it is not required to run the active app. The class-folder notation represents the 19 active checkpoint labels.

```text
FloraSense-AI (1)/                  Active project root
|-- app.py                         Streamlit UI and page routing
|-- styles.css                     Local UI theme
|-- model.py                       CNN checkpoint loader and inference engine
|-- dataset.py                     Taxonomy, image transforms, loading, splits, weighting
|-- ood_detector.py                Image screening and confidence/OOD decisions
|-- plant_data.py                  Plant profiles, aliases, and care lookup
|-- train_models.py                EfficientNet/MobileNet training pipeline
|-- evaluate_models.py             CNN checkpoint evaluation and plots
|-- train.py                       Separate ViT training path
|-- evaluate.py                    Separate ViT evaluator; root import is not verified
|-- download_real_dataset.py        Hugging Face archive preparation script
|-- collect_gbif_images.py          GBIF licensed-media candidate collector
|-- audit_dataset.py                Read-only image-tree audit utility
|-- backend/
|   `-- api.py                     Optional FastAPI endpoints
|-- checkpoints/
|   |-- efficientnet_b0_best.pt    Active primary CNN weights
|   |-- mobilenet_v3_small_best.pt Optional CNN weights
|   |-- class_mapping.json         Index-to-label mapping shared by CNN checkpoints
|   |-- preprocessing_config.json  Recorded input normalization configuration
|   |-- ood_thresholds.json        Stored calibration metadata
|   |-- training_metadata.json     EfficientNet production-run metadata
|   |-- efficientnet_b0_history.json
|   `-- mobilenet_v3_small_history.json
|-- data/
|   |-- train/<19 class folders>/  4,616 images in the saved audit
|   |-- val/<19 class folders>/    977 images
|   |-- test/<19 class folders>/   977 images
|   `-- dataset_manifest.json     Prepared-data manifest, claims archive source/license
|-- data_gbif_smoke/               30 GBIF candidate images; pending review
|   `-- source_manifest.json
|-- data_gbif_staging/             Partial GBIF collection; not used by training
|   `-- source_manifest.json       Incomplete relative to files currently in folders
|-- saved_models/florasense_vit/   Separate 47-label ViT artifact/configuration
|-- tests/
|   |-- test_dataset.py
|   |-- test_model.py
|   |-- test_ood_detector.py
|   |-- test_plant_data.py
|   `-- conftest.py
|-- FloraSense_Colab_Training.ipynb
|-- requirements.txt
|-- .env.example
|-- README.md
|-- evaluation_report.json         Saved CNN evaluation results
|-- dataset_audit_report.json      Saved audit report for `data/`
|-- dataset_manifest.json          Optional synthetic demo manifest; generated by demo code
|-- gbif_candidate_inventory.json  Occurrence-record inventory, not image counts
|-- house_plant_species.tar        Local source archive (5.27 GB)
|-- efficientnet_b0_confusion_matrix.png
|-- confusion_matrix.png           Generic/ViT evaluator artifact name
|-- model_comparison_confusion_matrix.png
|-- training_validation_curves.png
`-- .venv/ and caches              Environment and regenerable runtime files
```

### Important module relationships

- `app.py` imports `dataset.py`, `model.py`, `ood_detector.py`, and `plant_data.py`. It reads `checkpoints/class_mapping.json` for the catalog, `evaluation_report.json` for saved metrics, and `styles.css` for presentation.
- `model.py` imports `build_model` from `train_models.py` at engine initialization, loads a checkpoint and the adjacent class mapping, then returns ranked predictions.
- `train_models.py` uses `dataset.py` for loading, transforms, weights, splits, and mappings; it uses `ood_detector.py` to calibrate thresholds.
- `evaluate_models.py` loads `model.py` checkpoints and real test images using `dataset.py`, and writes JSON/plot artifacts.
- `backend/api.py` separately exposes health, species, care, and image-identification routes using the same model/data modules.
- `tests/` covers data helpers, model loading, image screening/OOD decisions, and care profiles. It does not automate browser navigation.

### Active, optional, and unverified paths

The active demonstration is the root `app.py` with EfficientNet-B0 by default. MobileNetV3-Small is also offered in the UI and has a checkpoint, but the saved evaluation report shown here contains an EfficientNet entry only. The root `train.py` is a Transformers ViT trainer, distinct from `train_models.py`. The root `evaluate.py` imports `PlantClassifier`, which is not defined in the root `model.py`; its compatibility with the root project is therefore not verified and it is expected to fail at import unless the appropriate alternate model implementation is used.

## 5. System Architecture

```mermaid
flowchart LR
    U[User in browser] --> UI[Streamlit app.py]
    UI --> UP[Image uploader and RGB decode]
    UP --> Q{Image quality check}
    Q -- unsuitable --> R[Show image-quality message]
    Q -- acceptable --> SEL[Selected CNN engine]
    SEL --> L[load_inference_engine]
    L --> CKPT[checkpoints/ARCH_best.pt]
    L --> MAP[checkpoints/class_mapping.json]
    SEL --> PRE[Resize 256, center crop 224, normalize]
    PRE --> INF[InferenceEngine.predict]
    INF --> SOFT[Softmax and ranked top-k labels]
    SOFT --> DEC{OOD enabled?}
    DEC -- yes --> OOD[Energy, confidence, and margin decision]
    DEC -- no --> CONF[Top confidence threshold only]
    OOD --> STATE[Certain / uncertain]
    CONF --> STATE
    STATE --> CARE[get_plant_profile]
    CARE --> VIEW[Prediction, alternatives, care tabs]
    STATE --> VIEW
```

The root application also has a separate optional FastAPI service. It is not a network backend used by Streamlit; the Streamlit app directly loads the local checkpoint. Plant profiles are looked up from Python dictionaries in `plant_data.py`.

## 6. End-to-End Image Identification Workflow

```mermaid
flowchart TD
    A[Choose JPG, JPEG, PNG, or WebP] --> B[Decode with Pillow and convert to RGB]
    B --> C{At least 50x50, usable exposure, sharp enough?}
    C -- no --> D[status=unsuitable; display quality message]
    C -- yes --> E[Load cached EfficientNet-B0 or selected MobileNet]
    E --> F[Resize to 256, center crop 224, tensor and ImageNet normalize]
    F --> G[Run CNN and produce logits]
    G --> H[Softmax; return up to 5 ranked candidates]
    H --> I{OOD filtering enabled?}
    I -- yes --> J[Energy score plus confidence and top-2 margin checks]
    I -- no --> K[Compare top-1 confidence with selected threshold]
    J --> L{Decision}
    K --> L
    L -- uncertain --> M[Show unconfirmed candidates; care is reference-only]
    L -- certain --> N[Show leading result and alternatives]
    N --> O[Lookup profile by label, common name, scientific name, or alias]
    O --> P[Render care and safety sections when profile exists]
    D --> Q[Stop; no prediction displayed]
```

The exact thresholds can be changed in the sidebar. When OOD is on, `calibrate_and_filter` marks a result uncertain if its energy is below the threshold or if confidence/margin checks fail. When OOD is off, the Streamlit path only checks the top prediction's confidence. The separate OOD helper also supports an `unsuitable` decision for an empty candidate list, but the UI's image-quality failure is handled before model inference.

The stored `ood_thresholds.json` reports temperature 1.35, energy threshold 1.68, confidence threshold 0.2695, and margin threshold 0.05, calibrated from 977 validation samples (784 correct predictions). The Streamlit energy slider is coded with a range of -40 to -5 even though the stored energy threshold is 1.68; the effective slider default should be reviewed. The app passes temperature 1.35 as a literal rather than reading the stored temperature field.

## 7. Machine-Learning Model

### CNN and EfficientNet-B0

A convolutional neural network (CNN) applies learned filters across local image regions to build increasingly abstract feature representations. Early layers respond to edges, colors, and textures; deeper layers combine them into higher-level shapes and patterns. EfficientNet is a family of convolutional architectures designed around scaling depth, width, and input resolution. This project uses Torchvision's `efficientnet_b0` builder rather than implementing the architecture itself.

EfficientNet-B0 is the documented primary baseline. The saved evaluation report lists approximately 4.03 million parameters and 24.48 MB checkpoint size. The current app defaults to it. The optional MobileNetV3-Small path is provided as a lighter CNN alternative.

### Transfer learning as implemented

`train_models.py` calls Torchvision's default pretrained weights for the selected CNN, replaces the classifier head with a project-specific output layer, and trains in two phases:

1. **Head warmup:** freeze the `features` backbone and train the classifier.
2. **Fine-tuning:** unfreeze the top two `features` stages and optimize backbone and head groups at different learning rates.

This differs from training from scratch: transfer learning starts from features learned on a large earlier dataset, then adapts the classifier and selected layers to plant images. Inference builds the same architecture with `pretrained=False` and loads the local trained state dictionary.

### Implemented training settings

| Setting | Source-code default / behavior |
|---|---|
| Architecture | `efficientnet_b0` by default; also `mobilenet_v3_small` |
| Input | 224 x 224 model crop; resize to 256 before center crop for validation/inference |
| Training augmentation | Random resized crop (scale 0.75-1.0, ratio 0.85-1.15), horizontal flip, rotation up to 15 degrees, color jitter 0.15 |
| Validation/test transform | Resize 256, center crop 224, tensor conversion, ImageNet mean/std normalization |
| ImageNet normalization | Mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]` |
| Loss | Weighted cross-entropy; inverse-frequency class weights normalized to mean 1 |
| Optimizer | AdamW, weight decay `1e-4` |
| Default learning rates | Head `5e-4`; backbone `1e-5`; fine-tuning head group uses half the head LR |
| Batch size | 32 |
| Epochs | 5 warmup + 15 fine-tuning by CLI defaults |
| Scheduler | Cosine annealing per phase |
| Early stopping | Patience 5, after at least 3 epochs in a phase |
| Gradient clipping | Norm 1.0 |
| Seed | 42 by default; CUDA/CuDNN deterministic flag set |
| Minimum class count | 10 by default, reported as a deficiency rather than silently balancing with synthetic data |

### Training and evaluation workflow

```mermaid
flowchart TD
    A[data directory] --> B{Split structure}
    B -- pre-split train/val/test --> C[Validate each split separately]
    B -- flat class folders --> D[Validate then seeded per-class split]
    C --> E[Train, validation, and test samples]
    D --> E
    E --> F[Training transforms and class weights]
    F --> G[Pretrained Torchvision CNN with replacement classifier]
    G --> H[Phase 1: freeze feature backbone]
    H --> I[Validate top-1 and top-3]
    I --> J[Phase 2: unfreeze top two feature stages]
    J --> K[Validate each epoch; early stop on stale validation score]
    K --> L[Save best validation top-1 checkpoint]
    L --> M[Write mapping, preprocessing config, history, metadata]
    K --> N[Calibrate OOD thresholds on validation loader]
    T[data/test] --> O[evaluate_models.py]
    L --> O
    O --> P[Saved metrics, per-class report, confusion matrix]
```

For the current dataset, `create_real_splits()` detects the already-separated directories and loads them. Training uses the training and validation samples; the separate evaluator reads the held-out test tree. The split code does not group by source identity, so the flow does not imply leakage-free splits.

These are the current script defaults. The exact hyperparameters used to create the currently installed production checkpoint are **not verified**: `training_metadata.json` marks the run production but omits warmup/fine-tune epochs; `efficientnet_b0_history.json` contains only one epoch in each phase and a `best_acc` of 0.0833, inconsistent with checkpoint metadata's validation top-1 of about 0.8025. Treat the history as a possibly stale artifact, not proof of this checkpoint's training trajectory.

### Checkpoint and class mapping

The root app's default loader calls `load_inference_engine('./checkpoints', 'efficientnet_b0')`, which opens `checkpoints/efficientnet_b0_best.pt` and `checkpoints/class_mapping.json`. The checkpoint contains the state dictionary and training values; the shared mapping binds integer output indices to the exact label strings. The active mapping has 19 classes. The recorded `preprocessing_config.json` stores 224 crop/input size, 256 resize size, bilinear interpolation, and ImageNet normalization values.

The active checkpoint SHA-256 observed on 2026-10-18 is:

```text
AE9AC1A0491538771C7D2660D5BA863B54B901775B8B50245578558B11CBEF8A
```

Do not copy a checkpoint without its matching `class_mapping.json`; changed class order would change the meaning of prediction indices.

## 8. Dataset Documentation

### Active training/evaluation data

`checkpoints/training_metadata.json` records `data_dir: ./data`, `is_demo_run: false`, and `status: trained_production` for EfficientNet-B0. `data/dataset_manifest.json` identifies `kakasher/house-plant-species`, claims CC BY 3.0, warns that images came from Bing Images and non-expert curation, and records 6,570 processed images. Rattlesnake Plant remains excluded because the archive's 315 images for that label are exact duplicates of Calathea images. Per-image origin/licensing records are not available.

The current `data/` tree has 19 class directories per split. Audited image totals:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| African Violet | 237 | 50 | 50 | 337 |
| Aloe Vera | 177 | 37 | 37 | 251 |
| Areca Palm | 133 | 28 | 28 | 189 |
| Boston Fern | 216 | 45 | 45 | 306 |
| Calathea | 453 | 96 | 96 | 645 |
| Chinese Evergreen | 360 | 77 | 77 | 514 |
| Dracaena | 183 | 39 | 39 | 261 |
| English Ivy | 168 | 36 | 36 | 240 |
| Jade Plant | 249 | 52 | 52 | 353 |
| Monstera Deliciosa | 383 | 82 | 82 | 547 |
| Orchid | 164 | 35 | 35 | 234 |
| Peace Lily | 271 | 57 | 57 | 385 |
| Pothos | 171 | 36 | 36 | 243 |
| Prayer Plant | 280 | 60 | 60 | 400 |
| Rubber Plant | 205 | 43 | 43 | 291 |
| Snake Plant | 278 | 59 | 59 | 396 |
| Tradescantia | 239 | 51 | 51 | 341 |
| Venus Flytrap | 141 | 29 | 29 | 199 |
| ZZ Plant | 308 | 65 | 65 | 438 |
| **Total** | **4,616** | **977** | **977** | **6,570** |

The total split is approximately 70.3% train / 14.9% validation / 14.9% test. Areca Palm is the smallest class (189); Calathea is the largest (645), a maximum-to-minimum ratio of 3.413. Neither the active dataset nor model mapping contains a separate Rattlesnake Plant class.

### Cleaning, augmentation, and duplicate limitations

- `download_real_dataset.py` verifies decodability, minimum dimensions of 50 pixels, computes MD5 hashes of source archive entries for exact deduplication, and writes class folders into seeded train/validation/test splits. Its defaults use 15% validation and 15% test.
- The downloader's `SELECTED_CLASSES` mapping contains 32 candidate labels (the original 19 plus 13 expansion candidates). Its `--add_class` option skips exact duplicates across existing splits and refuses to add a class if too few unique images remain. This archive has no unique Rattlesnake Plant images after that check. The default full preparation mode can overwrite classes in the target `--data_dir`.
- `dataset.py` verifies files, computes a 32 x 32 grayscale hash, reports duplicate counts, and computes per-class weights. Its current loader appends duplicate files to `samples`; its duplicate counter does not itself remove those samples. Pre-split folders are scanned separately, so the loader alone does not detect cross-split leakage.
- `audit_dataset.py` reports zero corrupt images, zero exact SHA-256 duplicate groups, 35 heuristic quality warnings, 1,653 near-duplicate pairs at average-hash distance <= 3, and 807 possible near-duplicate cross-split pairs. Near-duplicate flags are candidates for review, not confirmed identical images.
- `check_image_quality()` rejects dimensions below 50 x 50 in code. Its user-facing error text asks for at least 224 x 224, so the actual threshold and wording differ.
- The training transform performs random crop, horizontal flip, rotation, and color jitter. These are online tensor transforms; they do not create saved synthetic training images.
- There is no verified file-level source identity or same-source grouping in the active `data/` manifest. The same-source leakage status is therefore **not verified**.

### Dataset source and license limitations

The local README, `download_real_dataset.py`, and `data/dataset_manifest.json` label the archive CC BY 3.0. The current Hugging Face dataset card displays a CC BY 3.0 badge, but its text also states: “The current dataset is for personal use only due to copyright considerations.” Since the card's terms and code's blanket claim conflict, the right to redistribute or reuse all underlying Bing images commercially is **not verified**. Obtain permission/clarification before redistribution or non-personal use. The archive is present locally at `house_plant_species.tar` (5.27 GB); its presence does not prove legal rights.

### GBIF candidate data is not training data

- `data_gbif_smoke/` holds 30 images across 15 candidate species, with two images per species. Its `source_manifest.json` marks label and suitability review as pending.
- `data_gbif_staging/` currently contains 213 image files, but its manifest records only the Anthurium collection (94 manifest rows). A Birds Nest Fern folder and other small remnants do not have matching complete provenance entries. Do not use this staging tree for training without reconciling files and manifests.
- `gbif_candidate_inventory.json` counts license-filtered **occurrence records**, not image files or human-verified usable images.
- `collect_gbif_images.py` resolves exact GBIF species matches, requests occurrence records with still-image media, accepts per-media CC0 1.0 or CC BY 4.0, records attribution/license/source fields, rejects invalid or small images and exact duplicates in its output tree, and labels all images pending human review. The record count does not validate the photo's label.
- Neither GBIF folder is referenced by the production training metadata (`./data`). They are candidate/review material only.

### Root vs data manifest

`data/dataset_manifest.json` describes the prepared real-image tree. The optional root `dataset_manifest.json` is written by `create_synthetic_demo_splits()` and records 20 taxonomy labels with synthetic 40/20/20 train/validation/test sizes and 15 OOD images. It is demo metadata, not the active real dataset report, and can be deleted and regenerated by the demo helper or tests.

## 9. Plant-Care Information

`plant_data.py` defines Python dictionaries containing common/scientific names, family, descriptions, plant type, maintenance, location, sunlight, water, soil, temperature/humidity, pruning, features, toxicity/safety text, and sometimes references/aliases. `get_plant_profile()` first checks exact dictionary keys, then common/scientific names and aliases, and then a directional substring fallback.

The root Streamlit app requests a profile using the selected label or predicted common name. Care is shown fully for a `certain` result. For an `uncertain` result, the app labels the profile as an unconfirmed reference. Missing fields display an unavailable message. Toxicity text comes from the local profile; it is not queried from ASPCA or another service at runtime.

The module comments describe the profiles as cross-referenced with Kew POWO and RHS and some entries contain explicit source strings. Individual profile facts and every toxicity claim have **not** been independently revalidated for this documentation. The interface includes a safety disclaimer; it is not medical or veterinary advice.

## 10. Evaluation and Existing Results

`evaluate_models.py` evaluates available CNN checkpoints on `./data/test`, if present. It loads each engine, computes top-1/top-3 accuracy, `classification_report` per-class precision/recall/F1/support, confusion matrices, latency and parameter counts. It applies the saved OOD thresholds to test samples and counts uncertain decisions as false rejections. It writes `evaluation_report.json`, architecture confusion matrices such as `efficientnet_b0_confusion_matrix.png`, and `training_validation_curves.png` when history data exists.

The checked-in/current `evaluation_report.json` contains an existing EfficientNet-B0 report on **977 held-out test images**. This documentation did not rerun that evaluation:

| Metric | Existing report |
|---|---:|
| Top-1 accuracy | 81.88% |
| Top-3 accuracy | 94.58% |
| Macro-F1 | 81.49% |
| Weighted-F1 | 81.67% |
| Recorded inference latency | 21.17 ms/image |
| Parameter count | 4.03 million |
| Checkpoint size | 24.48 MB in the report |
| Uncertain/false-rejection rate | 14.23% |
| False-acceptance rate | `null` / not reported |
| Test support | 977 |

Top-1 accuracy is the fraction whose highest-scoring label matches the test label. Top-3 counts a correct label among the first three. Precision is the fraction of a predicted class's positive predictions that are correct; recall is the fraction of the true class recovered; F1 is their harmonic mean. Macro-F1 averages class F1 values equally, so small classes count as much as large classes. Weighted-F1 weights each class by support. The confusion matrix counts true-vs-predicted labels.

Per-class values copied from the existing classification report:

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| African Violet | 97.8% | 88.0% | 92.6% | 50 |
| Aloe Vera | 81.6% | 83.8% | 82.7% | 37 |
| Areca Palm | 73.5% | 89.3% | 80.6% | 28 |
| Boston Fern | 93.0% | 88.9% | 90.9% | 45 |
| Calathea | 84.8% | 81.3% | 83.0% | 96 |
| Chinese Evergreen | 78.7% | 76.6% | 77.6% | 77 |
| Dracaena | 59.6% | 87.2% | 70.8% | 39 |
| English Ivy | 85.3% | 80.6% | 82.9% | 36 |
| Jade Plant | 77.4% | 92.3% | 84.2% | 52 |
| Monstera Deliciosa | 81.2% | 84.1% | 82.6% | 82 |
| Orchid | 85.0% | 97.1% | 90.7% | 35 |
| Peace Lily | 86.0% | 64.9% | 74.0% | 57 |
| Pothos | 68.4% | 36.1% | 47.3% | 36 |
| Prayer Plant | 78.8% | 86.7% | 82.5% | 60 |
| Rubber Plant | 88.1% | 86.0% | 87.1% | 43 |
| Snake Plant | 88.7% | 79.7% | 83.9% | 59 |
| Tradescantia | 72.3% | 92.2% | 81.0% | 51 |
| Venus Flytrap | 100.0% | 82.8% | 90.6% | 29 |
| ZZ Plant | 86.7% | 80.0% | 83.2% | 65 |

These are historical saved values, not guaranteed future behavior. Test-set images may be correlated with training sources; the dataset audit flags possible near-duplicate leakage. Pothos recall/F1 is low in the report. `false_acceptance_rate_pct` is `null`; the evaluator has an `ood_dir` argument, but the inspected evaluation loop does not consume that directory or calculate false acceptance on a separate OOD set. No MobileNet metrics are in this report.

The README labels the recorded latency as CPU latency, but the JSON report does not record the device; the evaluator measures on the device selected by `InferenceEngine`. The actual hardware used for that timing is therefore **not verified in the report**.

The root `evaluate.py` is a separate ViT evaluator that imports `PlantClassifier`; that class is absent in the root `model.py`. It is not the CNN evaluator and is not verified as runnable from the root project.

## 11. Testing and Quality Assurance

| Test file | Main checks |
|---|---|
| `tests/test_dataset.py` | Taxonomy presence, class mappings, transforms, class weights, dataset wrapper, missing-directory error, synthetic smoke split |
| `tests/test_model.py` | Available checkpoint diagnostics, load behavior, prediction tensor/output structure, missing checkpoint behavior, species metadata alias |
| `tests/test_ood_detector.py` | Image quality accept/reject, energy score, confidence/uncertainty decisions, fallback thresholds |
| `tests/test_plant_data.py` | Care profile lookups, toxic/safe text, separation of Snake Plant/Rattlesnake Plant, profile coverage |
| `tests/conftest.py` | Textured synthetic image fixture and checkpoint directory fixture |

The recorded test run after the UI update was `35 passed`. Some tests use synthetic fixture/demo images for code-path checks; those are not training images or model-accuracy evidence. `test_model.py` can skip prediction-shape testing if the checkpoint is absent, though the current root checkpoint exists.

## 12. Setup and Execution Guide

Commands below are for Windows PowerShell from the project root. This workspace already has `.venv`; do not recreate it if preserving the current environment.

### Create and activate a new environment

```powershell
Set-Location "C:\Users\Hp\Downloads\FloraSense-AI (1)"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python launcher is unavailable, install a compatible Python runtime first. The repository does not pin an exact interpreter version; the inspected workspace used Python 3.13.9.

### Environment configuration

`.env.example` lists optional names such as `CHECKPOINT_DIR`, `DEFAULT_ARCH`, and threshold/port values. A search of root Python source found no `python-dotenv`, `.env` loader, or environment-variable reads for these options. Treat the file as a reference template only; these variables do not currently configure the app. No API key is required by the local inference path.

### Start the app

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`. The verified workspace command using its configured interpreter was:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
```

### Tests, audit, and evaluation

```powershell
python -m pytest tests -q
python audit_dataset.py --data_dir ./data --output dataset_audit_report.json
python evaluate_models.py --checkpoint_dir ./checkpoints --test_dir ./data/test --output evaluation_report.json
```

`audit_dataset.py` writes a JSON report; it does not alter images. Evaluation writes report/plot files and therefore may replace existing outputs.

### Training (not required to run inference)

The defaults are shown below. Training writes model files and metadata to `./checkpoints`; use a separate output directory to protect the current weights:

```powershell
python train_models.py --arch efficientnet_b0 --data_dir ./data --warmup_epochs 5 --finetune_epochs 15 --batch_size 32 --head_lr 5e-4 --backbone_lr 1e-5 --checkpoint_dir ./checkpoints_review --seed 42
```

Do not run `--demo` against the production `./checkpoints` directory: the synthetic demo path writes class mapping/config/history and may save checkpoint weights there. Do not run the archive downloader over the existing `./data` until data rights and overwrite effects are resolved.

### Dataset/download utilities

The archive script's implemented command is:

```powershell
python download_real_dataset.py --data_dir ./data --min_images 50
```

It downloads/extracts a Hugging Face archive and writes train/validation/test folders. This archive's usage rights are unresolved as explained above. Back up real data and use a separate `--data_dir` before experimenting.

GBIF candidate inspection/collection commands are:

```powershell
python collect_gbif_images.py --inventory-only --inventory-output gbif_candidate_inventory.json
python collect_gbif_images.py --output-dir ./data_gbif_staging --target-images 500 --max-records 100000
```

The inventory reports occurrence-record counts, not usable image counts. Collected images are staged with pending review and must not be used as training data without reconciling their source manifest and human-verifying labels.

### Optional FastAPI service

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

The interactive API description is normally at `http://127.0.0.1:8000/docs`. This service is separate from Streamlit.

### Google Colab

`FloraSense_Colab_Training.ipynb` is an intended GPU workflow with dependency installation, repository clone, data preparation, EfficientNet/MobileNet training, evaluation, and checkpoint packaging. The notebook summary reports that none of its cells have been executed. Its clone URL is a `YOUR_USERNAME` placeholder, and it repeats the dataset-license claim that conflicts with the current card. It is therefore a template, not a verified end-to-end Colab run. Before using it, set a real repository URL and resolve dataset permissions. Select Runtime > Change runtime type > GPU, then run cells in order only after reviewing the cells that download data and write checkpoints.

## 13. Design Decisions and Security

- The Streamlit interface loads the model locally and describes uploads as local processing; it does not send images to a cloud inference API.
- `st.cache_resource` caches each selected model engine during the Streamlit process.
- `class_mapping.json` is required next to CNN weights to prevent index/label mismatch.
- The backend has permissive CORS (`allow_origins=["*"]` with credentials enabled). This is not a production security configuration; restrict origins and review authentication before exposing it.
- The backend reads the uploaded body before checking the 10 MB size limit. Do not expose it to untrusted traffic without reviewing upload/resource controls.
- `torch.load` loads local checkpoint files. Only load trusted checkpoints; do not accept model files from untrusted users.
- The `.env.example` file contains no secrets and is not read by current root code. No credentials are required for local inference.
- Dataset copyright and provenance limitations must be resolved before redistribution or broader deployment.
- The care database contains informational pet-safety text. It is not a substitute for veterinary/medical advice.

## 14. Limitations and Future Scope

### Confirmed repository limitations

- The active EfficientNet checkpoint has 19 classes, despite README text claiming 20.
- `dataset.py`'s default taxonomy has a duplicated Pothos alias entry and differs from the active checkpoint mapping (including the Areca Palm mismatch). The current Streamlit catalog reads the checkpoint mapping; other code such as the API's default taxonomy listing uses `get_class_mappings()` and may report a different set.
- `resolve_species_metadata()` uses the same default taxonomy/aliases; Areca Palm is absent from that taxonomy, so its predicted scientific-name display can fall back to the class label. The care database has an Areca profile, but that does not correct the model metadata helper.
- The prepared dataset's source attribution is not per-image. The source card's personal-use statement conflicts with the CC BY 3.0 badge and local manifests.
- The audit's 807 cross-split near-duplicate links are heuristic candidates. Their presence weakens confidence that train and test are fully independent.
- `efficientnet_b0_history.json` and production training metadata conflict; training hyperparameters for the installed checkpoint are not established by those artifacts.
- `data_gbif_staging/` is incomplete: 213 images are present but the manifest only records one species' 94 images. GBIF candidates are pending human label/suitability review and are not included in the active training run.
- Root `evaluate.py` refers to `PlantClassifier`, which does not exist in root `model.py`. The nested project copy was removed; the root app is the active project.
- The stored evaluation report provides no false-acceptance metric and no separate OOD set was evaluated by the inspected CNN loop.

### Possible future work (not currently implemented)

- Resolve image rights; add image-level source/licence records to all training data.
- Manually audit candidate labels, remove near-duplicates, and regroup splits by image/source identity before making new accuracy claims.
- Reconcile the taxonomy and metadata resolvers with the actual checkpoint mapping without silently changing class IDs.
- Produce a fresh, reproducible training history that records the exact code/config/data version and checkpoint hash.
- Add a genuinely independent OOD test set and report false acceptance/false rejection separately.
- Repair or retire the separate ViT evaluation path if ViT support remains in scope.
- Add automated Streamlit integration tests and API tests.

## 15. Viva and Guide-Review Preparation

### Short project summary

FloraSense AI is a local Streamlit demonstration that screens an uploaded plant image, classifies it among 19 classes using a trained EfficientNet-B0 checkpoint by default, applies confidence/OOD handling, and displays local plant-care references when available. It also includes optional MobileNet, training/evaluation scripts, a FastAPI service, and tests.

### Demonstration sequence

1. Start `python -m streamlit run app.py` and show the Home dashboard.
2. Open Identify Plant and upload a clear image from the supported test set.
3. Click Identify; discuss the predicted label, existing confidence output, and top alternatives.
4. Explain that an uncertain decision remains unconfirmed and should not be treated as fact.
5. Open the care tabs and point out the source database and safety disclaimer.
6. Open Supported Plants and explain that the active checkpoint mapping contains 19 labels.
7. Open Model Information and state that the displayed metrics are saved results on 977 test images, not a live benchmark.
8. If asked about the dataset, show the audit caveats and license conflict rather than claiming complete provenance.

### Likely viva questions

1. **What problem does FloraSense address?** It combines a constrained plant image classifier with a local care-reference display to make a project demonstration of identification and care lookup.
2. **What is the model's actual output space?** The active `class_mapping.json` has 19 labels; it is not the 20 labels claimed in some older documentation.
3. **Why use EfficientNet-B0?** It is the project's primary transfer-learning baseline, implemented through Torchvision, with a compact reported 4.03M-parameter model and measured inference results.
4. **What is transfer learning?** Reusing pretrained visual features and adapting a classifier/backbone to a narrower task, rather than learning all features from random initialization.
5. **What happens before classification?** Pillow decodes and converts the upload to RGB; the OOD helper checks minimum size, average brightness, and Laplacian-variance sharpness.
6. **How is the image transformed for inference?** Resize to 256, center crop to 224, convert to tensor, and normalize using ImageNet mean/std.
7. **What is a class mapping?** A fixed mapping from model output indices to class labels. It must match the checkpoint exactly.
8. **What does confidence mean?** It is the softmax score for a candidate. It is not a calibrated guarantee of correctness for every image.
9. **How is uncertainty determined?** When OOD filtering is enabled, the code checks energy, top confidence, and the gap to the second candidate. When disabled, the app checks top confidence only.
10. **What is the difference between validation and test data?** Validation is used during model development/calibration; test data is intended for final held-out assessment. The existing report uses 977 test images.
11. **What is macro-F1?** The arithmetic mean of per-class F1 scores, giving each class equal weight regardless of sample count.
12. **Why can top-3 accuracy exceed top-1?** A correct class can appear second or third even when the top-ranked class is wrong.
13. **What does the current report say?** It records 81.88% top-1, 94.58% top-3, and 81.49% macro-F1 on 977 held-out images. It is a saved result, not a new evaluation.
14. **Which class is a known weakness?** Pothos has reported recall 36.1% and F1 47.3%, showing difficulty distinguishing visually similar foliage in this dataset.
15. **Are there duplicate or leakage concerns?** Exact hash duplicates were zero in the audit, but 807 near-duplicate cross-split candidates were flagged heuristically; they need visual review.
16. **Was the full archive license verified?** No. The local manifest claims CC BY 3.0, while the dataset card also says personal use only. Resolve this before redistribution.
17. **Does the care database diagnose plant disease?** No. It displays species-level reference care and safety text only.
18. **Can the application identify any plant?** No. It is limited to the active 19 classes, and unseen plants may be marked uncertain if filtering is enabled.
19. **Is MobileNet part of the app?** Yes, it is an optional architecture selection with a checkpoint file; the checked-in evaluation report only contains EfficientNet results.
20. **What is not fully verified?** Exact training hyperparameters for the installed checkpoint, complete image rights, label correctness, same-source split independence, GBIF staging completeness, and the root ViT evaluation path.

## 16. Glossary and References

### Glossary

| Term | Meaning in this project |
|---|---|
| CNN | Convolutional neural network; a model that learns spatial image features using convolution operations. |
| EfficientNet-B0 | The Torchvision CNN architecture used as the primary active classifier. |
| Transfer learning | Reusing pretrained weights and adapting layers/classifier to the plant-label task. |
| Checkpoint | Saved model parameters plus training values, loaded by the inference engine. |
| Class mapping | JSON correspondence between output index and plant label. |
| Preprocessing | Image resize/crop/tensor/normalization steps before inference. |
| Augmentation | Random transformations applied to training images to vary training inputs. |
| Inference | Running a trained model on an input photo to compute logits and ranked scores. |
| Logit | Raw class score emitted by the model before softmax normalization. |
| Softmax confidence | A normalized model score for one class; not a guarantee of truth. |
| OOD | Out-of-distribution: an input that does not look sufficiently like the model's known training classes under the implemented checks. |
| Precision | Correct predictions for a class divided by all predictions made as that class. |
| Recall | Correct predictions for a class divided by all true instances of that class. |
| F1 | Harmonic mean of precision and recall. |
| Macro-F1 | Unweighted mean of per-class F1 values. |
| Support | Number of labeled test examples for a class. |
| Confusion matrix | Table counting true labels against predicted labels. |

### Project files inspected

`app.py`, `styles.css`, `model.py`, `dataset.py`, `ood_detector.py`, `plant_data.py`, `backend/api.py`, `train_models.py`, `evaluate_models.py`, `train.py`, `evaluate.py`, `download_real_dataset.py`, `collect_gbif_images.py`, `audit_dataset.py`, `requirements.txt`, `.env.example`, README, checkpoint JSON metadata, data manifests/audit report, evaluation report, test files, and the Colab notebook.

### External references

- [Hugging Face House Plant Species dataset card](https://huggingface.co/datasets/kakasher/house-plant-species) - card metadata/source description; its personal-use note conflicts with its CC BY 3.0 badge.
- [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/) - license text; does not resolve rights in underlying source images.
- [GBIF Occurrence API documentation](https://techdocs.gbif.org/en/openapi/v1/occurrence) and [GBIF terms](https://www.gbif.org/terms) - used by the candidate collector; per-media license/attribution still must be retained.
- [Torchvision EfficientNet documentation](https://docs.pytorch.org/vision/stable/models/efficientnet.html) - model builder reference.
- Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), ICML 2019.
- [Streamlit documentation](https://docs.streamlit.io/) - UI framework reference.
- [scikit-learn `classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) - precision, recall, F1, support, and averaging semantics.
- [Kew Plants of the World Online](https://powo.science.kew.org/) and [Royal Horticultural Society plant directory](https://www.rhs.org.uk/plants) - cited by the project care database; not queried or independently audited at runtime.
- [ASPCA toxic and non-toxic plants](https://www.aspca.org/pet-care/animal-poison-control/toxic-and-non-toxic-plants) - the code cites ASPCA in some profile text; this documentation does not independently validate each safety statement.

External references explain technologies or identify the stated source; the implementation facts and measurements above come from the inspected local repository artifacts unless marked otherwise.