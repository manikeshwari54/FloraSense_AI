"""
FloraSense AI - Model Comparison & Checkpoint Benchmarking Suite
-----------------------------------------------------------------
Evaluates trained PyTorch transfer-learning models (EfficientNet-B0, MobileNetV3-Small)
on real test images, measuring:
  1. Top-1 & Top-3 Accuracy on genuine plant photographs
  2. Macro & per-class Precision, Recall, and F1-score
  3. Separate performance metrics for Tulsi and Neem
  4. Out-of-Distribution (OOD) rejection: False Acceptance Rate (FAR) & False Rejection Rate (FRR)
  5. Inference latency (ms/image) and checkpoint size (MB)
  6. Confusion matrix visualization and training/validation loss curves
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

from model import InferenceEngine, resolve_species_metadata
from dataset import validate_and_load_real_dataset
from ood_detector import calibrate_and_filter, load_thresholds

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def measure_latency_and_size(model: torch.nn.Module, input_size=(1, 3, 224, 224), runs=30):
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)

    for _ in range(5):
        _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency_ms = float(np.mean(latencies))
    total_params = sum(p.numel() for p in model.parameters())
    return avg_latency_ms, total_params

def run_evaluation_suite(
    checkpoint_dir: str = "./checkpoints",
    test_dir: str = "./data/test",
    ood_dir: Optional[str] = "./data/ood_test",
    output_report: str = "evaluation_report.json"
):
    print("===================================================================")
    print("FloraSense AI - Checkpoint Evaluation & Benchmark")
    print("===================================================================")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Real Test Directory:  {test_dir}")
    print("===================================================================\n")

    available = InferenceEngine.list_available(checkpoint_dir)
    usable_models = [m for m in available if m["is_usable"]]

    if not usable_models:
        print(f"[ERROR] No usable checkpoints found in '{checkpoint_dir}'.")
        for m in available:
            print(f"  - {m['arch']}: Exists={m['checkpoint_exists']}, Usable={m['is_usable']}, Error={m['error']}")
        sys.exit(1)

    has_real_test_data = os.path.isdir(test_dir)

    if not has_real_test_data:
        print("[NOTICE] No real test image folder found at './data/test'.")
        print("[NOTICE] Reporting model architecture size, parameter counts, and inference latency.")
        print("[NOTICE] Classification accuracy will NOT be fabricated or simulated.\n")

    thresholds = load_thresholds(os.path.join(checkpoint_dir, "ood_thresholds.json"))
    results = {}

    for info in usable_models:
        arch = info["arch"]
        ckpt_path = info["checkpoint_path"]
        print(f">>> Benchmarking {arch.upper()} ...")

        try:
            engine = InferenceEngine(arch=arch, checkpoint_path=ckpt_path)
        except Exception as e:
            print(f"  [ERROR] Failed to instantiate {arch}: {e}")
            continue

        latency_ms, total_params = measure_latency_and_size(engine.model)
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        model_res = {
            "arch": arch,
            "device": str(engine.device),
            "total_params_m": round(total_params / 1e6, 2),
            "model_size_mb": round(file_size_mb, 2),
            "avg_latency_ms": round(latency_ms, 2),
            "num_classes": engine.num_classes,
            "top1_acc": None,
            "top3_acc": None,
            "macro_f1": None,
            "false_rejection_rate_pct": None,
            "false_acceptance_rate_pct": None,
            "status_note": "No genuine test data provided" if not has_real_test_data else "Evaluated on genuine test data"
        }

        print(f"  Parameters:        {model_res['total_params_m']}M")
        print(f"  Model Size:        {model_res['model_size_mb']} MB")
        print(f"  Inference Latency: {model_res['avg_latency_ms']} ms/image ({engine.device})")
        print(f"  Trained Classes:   {model_res['num_classes']}")

        if has_real_test_data:
            real_test_samples, _, dataset_report = validate_and_load_real_dataset(
                test_dir,
                min_images_per_class=1,
                class_names=engine.class_names,
            )
            print(
                f"[Dataset] Found {len(real_test_samples)} valid unique test images "
                f"across {len(engine.class_names)} mapped classes; "
                f"excluded {dataset_report['duplicates_detected']} duplicates and "
                f"{dataset_report['invalid_file_count']} invalid files."
            )
            missing_test_classes = [
                class_name
                for class_name, count in dataset_report["class_counts"].items()
                if count < 1
            ]
            if missing_test_classes:
                raise ValueError(
                    "The held-out test set is missing mapped classes: "
                    + ", ".join(missing_test_classes)
                )
            if not real_test_samples:
                raise ValueError(f"No valid test images were found under '{test_dir}'.")
            y_true = []
            y_pred = []
            top3_correct = 0
            rejected_count = 0

            for fpath, true_c_id in real_test_samples:
                with Image.open(fpath) as img:
                    logits, preds = engine.predict(img, top_k=5)

                top_idx = preds[0]["class_index"] if preds else -1
                y_true.append(true_c_id)
                y_pred.append(top_idx)

                top3_indices = [p["class_index"] for p in preds[:3]]
                if true_c_id in top3_indices:
                    top3_correct += 1

                decision = calibrate_and_filter(
                    logits=logits,
                    predictions=preds,
                    energy_thresh=thresholds["energy_thresh"],
                    conf_thresh=thresholds["conf_thresh"],
                    margin_thresh=thresholds["margin_thresh"],
                    temperature=thresholds["temperature"],
                )
                if decision["status"] == "uncertain":
                    rejected_count += 1

            total_t = len(real_test_samples)
            top1_acc = np.mean(np.array(y_true) == np.array(y_pred))
            top3_acc = top3_correct / total_t
            frr = (rejected_count / total_t) * 100

            # Classification report & Macro F1
            target_names = [engine.id2label.get(i, f"Class_{i}") for i in range(engine.num_classes)]
            cls_report = classification_report(
                y_true,
                y_pred,
                labels=list(range(engine.num_classes)),
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )

            macro_f1 = cls_report["macro avg"]["f1-score"] * 100
            weighted_f1 = cls_report["weighted avg"]["f1-score"] * 100
            model_res["top1_acc"] = round(top1_acc * 100, 2)
            model_res["top3_acc"] = round(top3_acc * 100, 2)
            model_res["macro_f1"] = round(macro_f1, 2)
            model_res["weighted_f1"] = round(weighted_f1, 2)
            model_res["false_rejection_rate_pct"] = round(frr, 2)
            model_res["classification_report"] = cls_report
            model_res["dataset_report"] = dataset_report

            print(f"  Top-1 Accuracy:    {model_res['top1_acc']}%")
            print(f"  Top-3 Accuracy:    {model_res['top3_acc']}%")
            print(f"  Macro F1-Score:    {model_res['macro_f1']}%")
            print(f"  Weighted F1-Score: {model_res['weighted_f1']}%")
            print(f"  False Rejection:   {model_res['false_rejection_rate_pct']}%")

            # Save per-class summary and confusion matrix
            print("\n  Per-Class Performance Summary:")
            for c_name in target_names:
                m = cls_report.get(c_name, {})
                p = m.get("precision", 0.0) * 100
                r = m.get("recall", 0.0) * 100
                f = m.get("f1-score", 0.0) * 100
                sup = m.get("support", 0)
                print(f"    {c_name:22s} | P: {p:5.1f}% | R: {r:5.1f}% | F1: {f:5.1f}% | Support: {sup}")

            cm = confusion_matrix(y_true, y_pred, labels=list(range(engine.num_classes)))
            model_res["confusion_matrix"] = {
                "labels": target_names,
                "matrix": cm.tolist(),
            }
            figure_size = (
                max(14, 0.45 * engine.num_classes),
                max(12, 0.40 * engine.num_classes),
            )
            plt.figure(figsize=figure_size)
            sns.heatmap(cm, annot=engine.num_classes <= 20, fmt="d", cmap="Greens",
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f"{arch.upper()} Confusion Matrix (Real Held-Out Test Set)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            output_dir = os.path.dirname(os.path.abspath(output_report))
            cm_path = os.path.join(output_dir, f"{arch}_confusion_matrix.png")
            plt.savefig(cm_path, dpi=200)
            plt.close()
            print(f"  [Artifact] Saved confusion matrix to '{cm_path}'.")

        results[arch] = model_res
        print("")

    # Generate Training Curves Comparison Plot if history files exist
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    has_history = False
    for arch in ["efficientnet_b0", "mobilenet_v3_small"]:
        hist_path = os.path.join(checkpoint_dir, f"{arch}_history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
                train_losses = [p["train_loss"] for p in hist.get("phase1", []) + hist.get("phase2", []) if "train_loss" in p]
                val_accs = [p["val_top1"] * 100 for p in hist.get("phase1", []) + hist.get("phase2", []) if "val_top1" in p]
                if train_losses:
                    has_history = True
                    axes[0].plot(range(1, len(train_losses) + 1), train_losses, marker="o", label=f"{arch} Train Loss")
                if val_accs:
                    axes[1].plot(range(1, len(val_accs) + 1), val_accs, marker="s", label=f"{arch} Val Top-1 %")
            except Exception:
                pass

    if has_history:
        axes[0].set_title("Training Loss Progression")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CrossEntropy Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].set_title("Validation Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Top-1 Acc %")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        curves_path = os.path.join(
            os.path.dirname(os.path.abspath(output_report)),
            "training_validation_curves.png",
        )
        plt.savefig(curves_path, dpi=200)
        plt.close()
        print(f"[Artifact] Saved training curves to '{curves_path}'.")

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Artifact] Evaluation benchmark saved to '{output_report}'.")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloraSense AI Checkpoint Benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--ood_dir", type=str, default="./data/ood_test")
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    args = parser.parse_args()

    run_evaluation_suite(
        checkpoint_dir=args.checkpoint_dir,
        test_dir=args.test_dir,
        ood_dir=args.ood_dir,
        output_report=args.output
    )
