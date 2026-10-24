"""
FloraSense AI - Vision Transformer (ViT) Fine-Tuning Pipeline
--------------------------------------------------------------
Fine-tunes a Vision Transformer (ViT-Base) on plant image datasets.
Supports local directory dataset or Hugging Face dataset.
"""

import os
import sys
import argparse
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup
)

from dataset import SPECIES_TAXONOMY, get_class_mappings, create_real_splits, IMAGENET_MEAN, IMAGENET_STD

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class HFDatasetWrapper(Dataset):
    """Wraps HuggingFace dataset splits to return PIL Image and integer label."""
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB") if isinstance(item["image"], Image.Image) else item["image"]
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

class SyntheticPlantDataset(Dataset):
    """Synthetic dataset strictly for pipeline smoke-testing only."""
    def __init__(self, num_samples=32, num_classes=49, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.new("RGB", (224, 224), color=(
            (idx * 37) % 255, (idx * 67) % 255, (idx * 97) % 255
        ))
        label = idx % self.num_classes
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            k = min(5, logits.size(1))
            _, top5_preds = torch.topk(logits, k=k, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    top5_acc = top5_correct / max(1, total)
    return epoch_loss, epoch_acc, top5_acc

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"\n=======================================================")
    print(f"[FloraSense AI] - ViT Training Pipeline")
    print(f"=======================================================")
    print(f"Device:           {device}")
    print(f"Base Architecture: {args.base_model}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Epochs:            {args.epochs}")
    print(f"Mode:              {'Demo / Synthetic Smoke-Test' if args.demo_run else 'Genuine Training'}")
    print(f"=======================================================\n")

    train_transform, val_transform = get_transforms()

    if args.demo_run:
        print("[WARNING] Running in --demo_run mode with synthetic data. Accuracy will NOT represent real performance.")
        target_classes = sorted(list(SPECIES_TAXONOMY.keys()))
        class_names, id2label, label2id = get_class_mappings(target_classes)
        train_ds = SyntheticPlantDataset(num_samples=32, num_classes=len(class_names), transform=train_transform)
        val_ds = SyntheticPlantDataset(num_samples=16, num_classes=len(class_names), transform=val_transform)
    elif os.path.exists(args.data_dir):
        print(f"Loading genuine dataset from '{args.data_dir}'...")
        from dataset import PlantImageFolderDataset
        train_samples, val_samples, test_samples, class_names, report = create_real_splits(args.data_dir)
        names, id2label, label2id = get_class_mappings(class_names)
        train_ds = PlantImageFolderDataset(train_samples, transform=train_transform)
        val_ds = PlantImageFolderDataset(val_samples, transform=val_transform)
    elif args.dataset_name:
        print(f"Streaming dataset '{args.dataset_name}' from Hugging Face...")
        from datasets import load_dataset
        raw_ds = load_dataset(args.dataset_name)
        hf_classes = raw_ds["train"].features["label"].names
        class_names, id2label, label2id = get_class_mappings(hf_classes)

        # 80/20 train/validation split
        split_ds = raw_ds["train"].train_test_split(test_size=0.2, seed=42)
        train_ds = HFDatasetWrapper(split_ds["train"], transform=train_transform)
        val_ds = HFDatasetWrapper(split_ds["test"], transform=val_transform)
    else:
        print("[ERROR] Neither --data_dir nor a valid dataset was found. Use --demo_run for smoke testing.")
        sys.exit(1)

    print(f"Configuring classification head with {len(class_names)} target classes...")
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Top-5: {val_top5*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            print(f"  --> Saved new best checkpoint to '{args.output_dir}'")

    print(f"\nViT Training completed. Best Val Accuracy: {best_val_acc*100:.2f}%")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloraSense AI ViT Training Script")
    parser.add_argument("--base_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./saved_models/florasense_vit")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--demo_run", action="store_true")

    args = parser.parse_args()
    run_training(args)
