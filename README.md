# FloraSense AI

FloraSense AI is a local plant-photo classification and care-reference application. The Streamlit interface screens an image, runs a trained CNN checkpoint, presents ranked plant candidates, and can display care information from a local catalog.

The original `checkpoints/` model still maps **19 plant classes** and remains unchanged. A separately trained EfficientNet-B0 is now available in `checkpoints_47/` with a verified 47-class mapping. The Streamlit model selector offers both checkpoints and defaults to the 47-class model when available. This is a constrained classifier, not a general botanical identifier, plant-disease diagnostic tool, or substitute for veterinary or medical advice.

## Features

- Streamlit UI for image identification, searchable supported classes, checkpoint-specific model performance, and plant-care references.
- EfficientNet-B0 · 47 classes is the default model; the original EfficientNet-B0 · 19 classes remains selectable as a fallback. Labels and class counts come from the selected checkpoint mapping.
- Unavailable or incompatible checkpoints remain on disk for diagnostics but are excluded from normal model selection.
- Image-quality screening and optional confidence/OOD-based uncertainty handling.
- A local plant-care catalog. Care profiles are independent of model classes; unavailable species return a clear fallback instead of unrelated guidance.
- Optional FastAPI service, training and evaluation scripts, dataset utilities, and pytest tests.

## Requirements and setup

Use Python with the dependencies in `requirements.txt`. From PowerShell at the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the workspace already has a `.venv`, activate and use it instead of recreating it.

## Run the application

Each selectable CNN needs its checkpoint and matching `class_mapping.json` under `checkpoints/` or `checkpoints_47/`. The app loads the mapping and preprocessing/OOD configuration from the selected checkpoint folder.

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

The UI uses a local cream, forest-green, sage, and coral theme. It does not download fonts or send uploaded images to an external inference service. The Model Performance page reads saved reports from the selected checkpoint folder; it does not run evaluation during app startup.

## Inference files and retained training assets

The app needs a usable model weight file, its `class_mapping.json`, and its matching preprocessing and OOD configuration in the same checkpoint directory. Training metadata and evaluation reports are optional for inference, but populate the model-information and performance pages. The current 47-class report and confusion-matrix image remain in `checkpoints_47/`.

The prepared `data/` and `data_47/` directories are not needed for ordinary Streamlit inference. They are intentionally retained for reproducibility, training, and evaluation. They can be archived outside the runtime project only after updating the paths passed to training, preparation, and evaluation scripts; no dataset or checkpoint files are removed by the app.

## Image preprocessing

Shared preprocessing functions are in `preprocessing/image_preprocessing.py`:

- **Training:** random resized crop, horizontal flip, rotation, and color jitter, followed by tensor conversion and normalization.
- **Validation, testing, and inference:** deterministic resize to 256, center crop to 224, tensor conversion, and ImageNet normalization.

Random augmentation is only used for training. The resize/crop settings and ImageNet mean/std match `checkpoints/preprocessing_config.json`; changing these values can make input tensors incompatible with the saved model's expected preprocessing.

## Run tests

```powershell
python -m pytest tests/test_dataset.py tests/test_model.py -q
python -m pytest -q
```

The tests use synthetic images for some checks. Passing tests validate code paths; they do not establish model accuracy.

## Train and evaluate

The 47-class workflow uses separate `data_47/` and `checkpoints_47/` folders, preserving the existing `data/` and `checkpoints/` assets. In this workspace, preparation has already produced `data_47/`; skip the first command below unless starting with a clean output directory. The preparer refuses to overwrite a non-empty destination. The source archive is not downloaded by these commands:

```powershell
python prepare_all_classes.py --archive .\house_plant_species.tar --output_dir .\data_47 --expected_classes 47 --min_images_per_class 10
python train_models.py --arch efficientnet_b0 --data_dir .\data_47 --checkpoint_dir .\checkpoints_47 --warmup_epochs 2 --finetune_epochs 3 --batch_size 8 --patience 2 --force_cpu
```

The archive preparer validates image decoding and dimensions, excludes exact decoded-pixel duplicates (including conflicting-label copies), reports class/image counts and invalid files, and creates seeded stratified train/validation/test directories. It refuses to overwrite a non-empty output folder. Training derives its class count and label mapping from those folders, saves the matching mapping/configuration/history/metadata, selects by validation performance, reloads the best checkpoint, then calibrates OOD thresholds using validation data only. The test set is reserved for evaluation. The completed 2-epoch warmup + 3-epoch fine-tuning run reached 77.93% validation top-1 and 91.13% validation top-3; these are validation metrics, not held-out test results. OOD thresholds were calibrated on all 2,198 validation images after reloading the best checkpoint.

The script defaults to 5 warmup epochs and 15 fine-tuning epochs, batch size 32. Run `python train_models.py --help` to see all options. Add `--force_cpu` when CPU execution is explicitly desired; otherwise CUDA is used when available. CPU training on the full dataset can be lengthy. `--demo` uses synthetic solid-color images for a smoke test, not for model-quality claims; use a separate checkpoint directory for demo runs as well.

Evaluate the saved CNN checkpoints using the test split:

```powershell
python evaluate_models.py --checkpoint_dir ./checkpoints --test_dir ./data/test --output evaluation_report.json
```

After a successful 47-class training run, evaluate only its separate held-out test split:

```powershell
python evaluate_models.py --checkpoint_dir .\checkpoints_47 --test_dir .\data_47\test --output .\checkpoints_47\evaluation_report.json
```

The completed 47-class held-out report on 2,198 test images records 76.84% top-1 accuracy, 90.99% top-3 accuracy, 76.21% macro-F1, and 76.73% weighted-F1. The Streamlit Model Performance page reads these saved metrics for the matching selected checkpoint. Full per-class precision/recall/F1/support and the labeled 47 x 47 confusion matrix are in `checkpoints_47/evaluation_report.json`; the plot is `checkpoints_47/efficientnet_b0_confusion_matrix.png`. These offline test metrics are not confidence scores for individual app predictions. The earlier scikit-learn `_cython_blas` Application Control denial did not reproduce during evaluation; no policy or package changes were made.

Evaluation can overwrite report and plot outputs. Back them up or choose a separate output path if you need to retain existing results.

## Existing evaluation report

The repository's saved `evaluation_report.json` records EfficientNet-B0 results on 977 test images:

| Metric | Saved value |
|---|---:|
| Top-1 accuracy | 81.88% |
| Top-3 accuracy | 94.58% |
| Macro-F1 | 81.49% |
| Weighted-F1 | 81.67% |
| Uncertain / false-rejection rate | 14.23% |

These are existing artifact values, not a new benchmark run. The report does not contain a false-acceptance rate, and the hardware used for the latency measurement is not recorded. Dataset near-duplicate candidates also limit confidence that train/test samples are fully independent. Pothos is a known weak class in this report (recall 36.1%, F1 47.3%). See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for metric interpretation and repository caveats.

## Dataset notes

The prepared dataset is under `data/train/`, `data/val/`, and `data/test/`. The local archive scan verified 47 source class folders and 14,790 decodable image files; exact decoded-RGB duplicate detection found no duplicate groups in that scan. The dataset card states personal use only due to copyright considerations, despite a conflicting license badge/local metadata. Rights and per-image provenance have not been independently resolved. Keep data and derived artifacts local unless reuse and redistribution rights are clarified. The existing `data/` and `checkpoints/` remain the original 19-class assets.

The saved audit reports possible near-duplicate images across splits. These are heuristic candidates and require review; they are not proof of leakage. Labels are not guaranteed to be botanically correct.

Dataset preparation utilities can download data and write into their target directories. Review their options and use a separate destination before running them:

```powershell
python download_real_dataset.py --help
python audit_dataset.py --data_dir ./data --output dataset_audit_report.json
```

For the verified archive layout, use `prepare_all_classes.py` as shown above rather than the older selected-class downloader.

GBIF-collected images are candidate material only. They require human label/suitability review and provenance reconciliation before training.

## Optional FastAPI service

By default the API loads from `checkpoints/`. To select the separately trained 47-class checkpoint, set the backend environment variable before startup:

```powershell
$env:FLORASENSE_CHECKPOINT_DIR = ".\checkpoints_47"
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs` while the service is running. This service is separate from the Streamlit app.

## Repository map

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `model.py` | CNN checkpoint loading and inference |
| `preprocessing/` | Shared training and deterministic evaluation/inference transforms |
| `dataset.py` | Taxonomy, dataset validation/loading, splits, and class weighting |
| `prepare_all_classes.py` | Read-only TAR audit and stratified preparation of discovered classes |
| `train_models.py` | EfficientNet-B0 and MobileNetV3-Small training |
| `evaluate_models.py` | CNN checkpoint evaluation and report/plot generation |
| `ood_detector.py` | Image-quality checks, confidence/OOD decisions, and threshold calibration |
| `plant_data.py` | Local plant-care profiles and lookup helpers |
| `backend/api.py` | Optional FastAPI service |
| `checkpoints/` | CNN checkpoints, class mapping, preprocessing config, and metadata |
| `tests/` | pytest suite |

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the detailed architecture, dataset and model caveats, setup notes, and existing artifact analysis.
