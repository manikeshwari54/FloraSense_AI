# 🌿 FloraSense AI — Indoor Plant Identification & Care

An AI-powered application that identifies 20 common indoor plant species and provides
verified botanical care guidance. Built with PyTorch transfer learning on the
[House Plant Species dataset](https://huggingface.co/datasets/kakasher/house-plant-species).

## Supported Species (20 Indoor Plants)

| # | Plant | Scientific Name | Pet Safety |
|---|-------|----------------|------------|
| 1 | African Violet | Streptocarpus ionanthus | ✅ Non-toxic |
| 2 | Aloe Vera | Aloe vera | ⚠️ Toxic to pets |
| 3 | Areca Palm | Dypsis lutescens | ✅ Non-toxic |
| 4 | Boston Fern | Nephrolepis exaltata | ✅ Non-toxic |
| 5 | Calathea | Goeppertia spp. | ✅ Non-toxic |
| 6 | Chinese Evergreen | Aglaonema commutatum | ⚠️ Toxic to pets |
| 7 | Dracaena | Dracaena spp. | ⚠️ Toxic to pets |
| 8 | English Ivy | Hedera helix | ⚠️ Toxic to pets |
| 9 | Jade Plant | Crassula ovata | ⚠️ Toxic to pets |
| 10 | Monstera Deliciosa | Monstera deliciosa | ⚠️ Toxic to pets |
| 11 | Orchid | Phalaenopsis spp. | ✅ Non-toxic |
| 12 | Peace Lily | Spathiphyllum wallisii | ⚠️ Toxic to pets |
| 13 | Pothos | Epipremnum aureum | ⚠️ Toxic to pets |
| 14 | Prayer Plant | Maranta leuconeura | ✅ Non-toxic |
| 15 | Rattlesnake Plant | Goeppertia lancifolia | ✅ Non-toxic |
| 16 | Rubber Plant | Ficus elastica | ⚠️ Toxic to pets |
| 17 | Snake Plant | Dracaena trifasciata | ⚠️ Toxic to pets |
| 18 | Tradescantia | Tradescantia spp. | ⚠️ Mildly toxic |
| 19 | Venus Flytrap | Dionaea muscipula | ✅ Non-toxic |
| 20 | ZZ Plant | Zamioculcas zamiifolia | ⚠️ Toxic to pets |

> **Dataset note:** Images sourced from Bing Images (non-expert curation). Label quality varies by class. Do not rely solely on this tool for safety decisions.

## Setup

### Windows (local)
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1  # PowerShell
# Or bypass execution policy:
powershell -ExecutionPolicy Bypass -File .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run the app
```powershell
python -m streamlit run app.py
```

## Download the Training Dataset

Download and prepare the House Plant Species dataset (14,790 images, CC BY 3.0):

```powershell
python download_real_dataset.py --data_dir ./data
```

This downloads ~150 MB from Hugging Face, extracts and selects 20 indoor classes, and creates `data/train/`, `data/val/`, `data/test/` splits (70/15/15).

**If the download fails** (network restrictions, Hugging Face access):
1. Go to https://huggingface.co/datasets/kakasher/house-plant-species
2. Download `house_plant_species.tar` manually
3. Run: `python download_real_dataset.py --tar_path ./house_plant_species.tar --skip_download`

## Train the Models

### EfficientNet-B0 (primary)
```powershell
python train_models.py --data_dir ./data --arch efficientnet_b0 --warmup_epochs 5 --finetune_epochs 15
```

### MobileNetV3-Small (lightweight)
```powershell
python train_models.py --data_dir ./data --arch mobilenet_v3_small --warmup_epochs 5 --finetune_epochs 15
```

### Demo mode (no real data — smoke test only, results not representative)
```powershell
python train_models.py --demo --warmup_epochs 2 --finetune_epochs 3
```

## Google Colab
See `FloraSense_Colab_Training.ipynb` for GPU training instructions.

## Measured Benchmark Results

All metrics evaluated on **977 held-out real test photographs** across 19 trained indoor species from the `kakasher/house-plant-species` dataset.

| Architecture | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Checkpoint Size | Latency (CPU) | False Rejection (OOD) |
|---|---|---|---|---|---|---|
| **EfficientNet-B0 (Primary)** | **81.88%** | **94.58%** | **81.49%** | 24.5 MB (4.03M params) | 21.2 ms/image | 14.23% |
| **MobileNetV3-Small (Edge)** | Deferred / Training | - | - | 14.3 MB | ~11 ms/image | - |

### Per-Class Test Performance (EfficientNet-B0)
- **Top performers**: Venus Flytrap (F1: 90.6%), African Violet (F1: 92.6%), Boston Fern (F1: 90.9%), Orchid (F1: 90.7%), Rubber Plant (F1: 87.1%), Jade Plant (F1: 84.2%).
- **Challenging foliage**: Pothos (F1: 47.3% — visual confusion with trailing Philodendron/Calathea patterns in web-scraped data).
- **Confusion Matrix**: Saved as `efficientnet_b0_confusion_matrix.png`.
- **Rejection & Safety**: Calibrated Helmholtz Energy threshold (`1.68`) and confidence threshold (`26.95%`) safely flag corrupt, blurry, and out-of-distribution non-plant images as `uncertain` or `unsuitable`.

## Evaluate
```powershell
python evaluate_models.py --checkpoint_dir ./checkpoints --test_dir ./data/test
```

Generates `evaluation_report.json`, confusion matrices, and training curves.

## Dataset Citation
```
Gregorizwicz, K. (2024). House Plant Species Dataset.
Hugging Face Datasets. https://huggingface.co/datasets/kakasher/house-plant-species
License: CC BY 3.0
```

## Dataset Limitations
- Images collected from Bing Images (not botanically verified)
- Label quality varies by class — some images may be mislabelled
- Only 20 of the 47 classes are used (outdoor/seasonal species excluded)
- Model trained on this data should be used as a guide, not a definitive identification tool

## Project Structure
```
FloraSense-AI/
├── app.py                      # Streamlit web application
├── model.py                    # InferenceEngine (EfficientNet/MobileNet)
├── dataset.py                  # Dataset pipeline, transforms, class mappings
├── train_models.py             # Transfer learning training pipeline
├── evaluate_models.py          # Evaluation metrics and confusion matrices
├── ood_detector.py             # Out-of-distribution detection
├── plant_data.py               # Botanical care database (20 species)
├── download_real_dataset.py    # Dataset downloader from Hugging Face
├── checkpoints/                # Saved model checkpoints + class mappings
├── data/                       # Training data (created by download script)
│   ├── train/
│   ├── val/
│   └── test/
├── backend/api.py              # FastAPI REST backend
└── tests/                      # pytest test suite
```
