"""
FloraSense AI - Transfer Learning Training & Fine-Tuning Pipeline
-----------------------------------------------------------------
Supports:
  1. EfficientNet-B0 (Primary Baseline)
  2. MobileNetV3-Small (Lightweight Edge Comparison)

Features:
  - Genuine image dataset ingestion with verification & deduplication
  - Two-Phase Transfer Learning (Head warmup -> Backbone fine-tuning)
  - Class-weighted Cross-Entropy loss for class imbalance
  - AdamW optimizer with Cosine Annealing learning rate scheduler
  - Early stopping & best validation checkpointing
  - Checkpoint integrity: saves class_mapping.json, preprocessing_config.json,
    and training_metadata.json alongside each checkpoint
  - Automatic OOD threshold calibration on validation data
"""

import os
import sys
import json
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from dataset import (
    get_class_mappings,
    get_data_transforms,
    PlantImageFolderDataset,
    compute_class_weights,
    create_real_splits,
    create_synthetic_demo_splits,
    check_neem_tulsi_coverage,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from ood_detector import calibrate_thresholds_on_validation

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """
    Builds model and replaces the classification head with exact num_classes.
    """
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. Choose 'efficientnet_b0' or 'mobilenet_v3_small'.")

    return model

def freeze_backbone(model: nn.Module, arch: str):
    """Freezes the feature extraction backbone."""
    if arch in ["efficientnet_b0", "mobilenet_v3_small"]:
        for param in model.features.parameters():
            param.requires_grad = False
    print(f"[{arch.upper()}] Feature backbone frozen for head warmup.")

def unfreeze_top_backbone(model: nn.Module, arch: str, num_stages_to_unfreeze: int = 2):
    """Selectively unfreezes top feature layers for fine-tuning."""
    if arch in ["efficientnet_b0", "mobilenet_v3_small"]:
        total_stages = len(model.features)
        for idx, stage in enumerate(model.features):
            if idx >= total_stages - num_stages_to_unfreeze:
                for param in stage.parameters():
                    param.requires_grad = True
            else:
                for param in stage.parameters():
                    param.requires_grad = False
    print(f"[{arch.upper()}] Top {num_stages_to_unfreeze} backbone stages unfrozen for fine-tuning.")

def evaluate_metrics(model, dataloader, criterion, device):
    """Calculates loss, top-1 accuracy, and top-3 accuracy."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, preds = torch.max(logits, 1)
            correct_top1 += (preds == labels).sum().item()

            max_k = min(3, logits.size(1))
            _, top3_preds = torch.topk(logits, k=max_k, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top3_preds[i]:
                    correct_top3 += 1

    avg_loss = running_loss / max(1, total)
    top1_acc = correct_top1 / max(1, total)
    top3_acc = correct_top3 / max(1, total)
    return avg_loss, top1_acc, top3_acc

def train_phase(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs, device, arch, phase_name, checkpoint_dir, best_acc=0.0, patience=5
):
    history = []
    print(f"\n--- Starting {phase_name} ({epochs} epochs) ---")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if scheduler:
            scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_top1, val_top3 = evaluate_metrics(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"[{phase_name}] Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1*100:.1f}% | Val Top-3: {val_top3*100:.1f}%")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top3": val_top3
        }
        history.append(record)

        if val_top1 > best_acc:
            best_acc = val_top1
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"{arch}_best.pt")
            torch.save({
                "arch": arch,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to '{checkpoint_path}' (Val Top-1: {best_acc*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 3:
                print(f"  --> Early stopping triggered after {patience} epochs without improvement.")
                break

    return best_acc, history

def run_training_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("===================================================================")
    print(f"FloraSense AI - Transfer Learning Training: {args.arch.upper()}")
    print("===================================================================")
    print(f"Device:           {device}")
    print(f"Architecture:     {args.arch}")
    print(f"Mode:             {'DEMO (Synthetic Verification)' if args.demo else 'PRODUCTION (Real Images)'}")
    print(f"Data Source:      {args.data_dir if not args.demo else 'Synthetic In-Memory Benchmark'}")
    print(f"Warmup Epochs:    {args.warmup_epochs} (Head only)")
    print(f"Fine-Tune Epochs: {args.finetune_epochs} (Backbone unfrozen)")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Checkpoint Dir:   {args.checkpoint_dir}")
    print("===================================================================\n")

    # 1. Dataset Loading & Split Preparation
    if args.demo:
        print("[WARNING] Running in --demo mode with synthetic solid-color image patches.")
        print("[WARNING] This verifies code execution only. Do NOT use for accuracy claims.\n")
        train_samples, val_samples, test_samples, ood_samples, manifest = create_synthetic_demo_splits()
        class_names = manifest["class_names"]
        is_demo_run = True
    else:
        if not os.path.exists(args.data_dir):
            print(f"\n[ERROR] Dataset directory '{args.data_dir}' does not exist.")
            print("Please create the directory and place your real plant photos organized by class:")
            print(f"  {args.data_dir}/<ClassName>/image.jpg")
            print("Or with pre-split folders:")
            print(f"  {args.data_dir}/train/<ClassName>/image.jpg")
            print(f"  {args.data_dir}/val/<ClassName>/image.jpg")
            print(f"  {args.data_dir}/test/<ClassName>/image.jpg")
            print("\nIf you want to run a pipeline verification check without local images, pass --demo.")
            sys.exit(1)

        try:
            train_samples, val_samples, test_samples, class_names, split_rep = create_real_splits(
                args.data_dir, min_images_per_class=args.min_images, seed=args.seed
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load dataset from '{args.data_dir}': {e}")
            sys.exit(1)
        is_demo_run = False

    num_classes = len(class_names)
    names, id2label, label2id = get_class_mappings(class_names)

    # Coverage Audit for Tulsi and Neem
    coverage = check_neem_tulsi_coverage(class_names)
    print("--- Species Coverage Diagnostics ---")
    print(f"Total Classes:  {num_classes}")
    print(f"Tulsi Present:  {'[OK] Yes' if coverage['tulsi_present'] else '[MISSING] No'}")
    print(f"Neem Present:   {'[OK] Yes' if coverage['neem_present'] else '[MISSING] No'}")
    if coverage["missing"]:
        print(f"Missing Key Species: {', '.join(coverage['missing'])}")
    print("------------------------------------\n")

    # Save Class Mapping (ensures strict synchronization between checkpoint and labels)
    mapping_file = os.path.join(args.checkpoint_dir, "class_mapping.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    # Save Preprocessing Configuration
    prep_file = os.path.join(args.checkpoint_dir, "preprocessing_config.json")
    with open(prep_file, "w", encoding="utf-8") as f:
        json.dump({
            "input_size": [224, 224],
            "crop_size": 224,
            "resize_size": 256,
            "interpolation": "bilinear",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        }, f, indent=2)

    train_transform, val_transform = get_data_transforms()
    train_dataset = PlantImageFolderDataset(train_samples, transform=train_transform)
    val_dataset = PlantImageFolderDataset(val_samples, transform=val_transform)

    train_labels = [label for _, label in train_samples]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = build_model(args.arch, num_classes=num_classes, pretrained=True).to(device)

    best_acc = 0.0

    # Resume checkpoint if requested
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.arch}_best.pt")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint '{checkpoint_path}'...")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_acc = ckpt.get("best_acc", 0.0)
            print(f"Resumed with previous best accuracy: {best_acc*100:.1f}%")

    # Phase 1: Warmup Classifier Head
    if args.warmup_epochs > 0:
        freeze_backbone(model, args.arch)
        optimizer_head = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.head_lr,
            weight_decay=1e-4
        )
        scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=args.warmup_epochs)
        best_acc, history_phase1 = train_phase(
            model, train_loader, val_loader, criterion, optimizer_head, scheduler_head,
            epochs=args.warmup_epochs, device=device, arch=args.arch,
            phase_name="Phase 1 (Head Warmup)", checkpoint_dir=args.checkpoint_dir,
            best_acc=best_acc, patience=args.patience
        )
    else:
        history_phase1 = []

    # Phase 2: Backbone Fine-Tuning
    if args.finetune_epochs > 0:
        unfreeze_top_backbone(model, args.arch, num_stages_to_unfreeze=2)
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer_ft = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr * 0.5}
        ], weight_decay=1e-4)

        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        best_acc, history_phase2 = train_phase(
            model, train_loader, val_loader, criterion, optimizer_ft, scheduler_ft,
            epochs=args.finetune_epochs, device=device, arch=args.arch,
            phase_name="Phase 2 (Backbone Fine-Tuning)", checkpoint_dir=args.checkpoint_dir,
            best_acc=best_acc, patience=args.patience
        )
    else:
        history_phase2 = []

    # Save Training History and Metadata
    history_file = os.path.join(args.checkpoint_dir, f"{args.arch}_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({"phase1": history_phase1, "phase2": history_phase2, "best_acc": best_acc}, f, indent=2)

    metadata_file = os.path.join(args.checkpoint_dir, "training_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "arch": args.arch,
            "num_classes": num_classes,
            "class_names": class_names,
            "has_tulsi": coverage["tulsi_present"],
            "has_neem": coverage["neem_present"],
            "best_acc": best_acc,
            "is_demo_run": is_demo_run,
            "data_dir": args.data_dir if not is_demo_run else "synthetic_demo",
            "warmup_epochs": args.warmup_epochs,
            "finetune_epochs": args.finetune_epochs
        }, f, indent=2)

    # Calibrate OOD threshold on validation loader
    try:
        calibrate_thresholds_on_validation(
            model, val_loader, device=device,
            save_path=os.path.join(args.checkpoint_dir, "ood_thresholds.json")
        )
        print("[OOD Engine] Empirical rejection thresholds calibrated on validation set.")
    except Exception as e:
        print(f"[OOD Engine Notice] Could not calibrate OOD thresholds: {e}")

    print(f"\nTraining completed for {args.arch.upper()}! Best Validation Top-1: {best_acc*100:.1f}%\n")
    return model, best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloraSense AI Transfer Learning Training")
    parser.add_argument("--arch", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to genuine dataset directory")
    parser.add_argument("--demo", action="store_true", help="Run demonstration smoke-test with synthetic data")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Classifier warmup epochs")
    parser.add_argument("--finetune_epochs", type=int, default=15, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--head_lr", type=float, default=5e-4, help="Classifier head learning rate")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_images", type=int, default=10, help="Minimum images per class for real data")

    args = parser.parse_args()
    run_training_pipeline(args)
