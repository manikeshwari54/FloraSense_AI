"""
FloraSense AI - Vision Transformer (ViT) Real Evaluation Suite
--------------------------------------------------------------
Evaluates the actual trained Vision Transformer classifier on genuine test photographs.
Random, simulated, or fabricated metrics are strictly eliminated.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import PlantClassifier
from dataset import validate_and_load_real_dataset, get_class_mappings

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def run_evaluation(test_dir: str = "./data/test", output_report: str = "evaluation_vit_report.json"):
    print("=======================================================")
    print("FloraSense AI - ViT Real Performance Evaluation")
    print("=======================================================")
    print(f"Test Directory: {test_dir}")
    print("=======================================================\n")

    if not os.path.exists(test_dir):
        print(f"[ERROR] Test directory '{test_dir}' was not found.")
        print("To run genuine evaluation, provide real test images in:")
        print(f"  {test_dir}/<ClassName>/image.jpg")
        print("\nNote: Simulated or fabricated accuracy metrics are strictly forbidden.")
        sys.exit(1)

    try:
        samples, test_classes, load_rep = validate_and_load_real_dataset(test_dir, min_images_per_class=1)
    except Exception as e:
        print(f"[ERROR] Failed to load test images: {e}")
        sys.exit(1)

    if not samples:
        print(f"[ERROR] No valid test photographs found in '{test_dir}'.")
        sys.exit(1)

    print(f"Loaded {len(samples)} real test photographs across {len(test_classes)} classes.")

    # Initialize actual ViT model
    print("Loading Dual Vision Transformer classifier...")
    try:
        classifier = PlantClassifier.get_instance()
    except Exception as e:
        print(f"[ERROR] Could not load ViT model: {e}")
        sys.exit(1)

    info = classifier.get_info()
    print(f"Model:        {info['architecture']}")
    print(f"Device:       {info['device']}")
    print(f"Total Params: {info['total_parameters']}\n")

    y_true_names = []
    y_pred_names = []
    y_true_indices = []
    y_pred_indices = []

    class_names, id2label, label2id = get_class_mappings(test_classes)
    top1_correct = 0
    top3_correct = 0

    latencies = []

    print("Running inference on test dataset...")
    for idx, (fpath, c_id) in enumerate(samples, 1):
        true_name = id2label[c_id]
        with Image.open(fpath) as img:
            rgb_img = img.convert("RGB")

        t0 = time.perf_counter()
        _, predictions = classifier.predict(rgb_img, top_k=5)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        pred_name = predictions[0]["common_name"] if predictions else "Unknown"
        top3_names = [p["common_name"].lower() for p in predictions[:3]]

        y_true_names.append(true_name)
        y_pred_names.append(pred_name)

        # Top-1 check
        is_top1 = (true_name.lower() in pred_name.lower()) or (pred_name.lower() in true_name.lower())
        if is_top1:
            top1_correct += 1

        # Top-3 check
        is_top3 = any(true_name.lower() in cand or cand in true_name.lower() for cand in top3_names)
        if is_top3:
            top3_correct += 1

        if idx % 20 == 0 or idx == len(samples):
            print(f"  Processed [{idx}/{len(samples)}] photos...")

    total = len(samples)
    top1_acc = top1_correct / total
    top3_acc = top3_correct / total
    avg_latency = float(np.mean(latencies))

    print("\n--- Genuine Evaluation Results ---")
    print(f"Total Test Images: {total}")
    print(f"Top-1 Accuracy:    {top1_acc*100:.2f}%")
    print(f"Top-3 Accuracy:    {top3_acc*100:.2f}%")
    print(f"Average Latency:   {avg_latency:.2f} ms/image")

    # Specific metrics for Tulsi and Neem
    def compute_class_subset_metrics(name_keyword):
        indices = [i for i, name in enumerate(y_true_names) if name_keyword.lower() in name.lower()]
        if not indices:
            return {"sample_count": 0, "correct_count": 0, "accuracy": None}
        correct = sum(1 for i in indices if name_keyword.lower() in y_pred_names[i].lower())
        return {
            "sample_count": len(indices),
            "correct_count": correct,
            "accuracy": round((correct / len(indices)) * 100, 2)
        }

    tulsi_stats = compute_class_subset_metrics("Tulsi")
    neem_stats = compute_class_subset_metrics("Neem")

    print(f"\nTulsi Statistics: {tulsi_stats['correct_count']}/{tulsi_stats['sample_count']} correct "
          f"({tulsi_stats['accuracy']}% accuracy)" if tulsi_stats['sample_count'] else "\nTulsi Statistics: Not present in test folder")
    print(f"Neem Statistics:  {neem_stats['correct_count']}/{neem_stats['sample_count']} correct "
          f"({neem_stats['accuracy']}% accuracy)" if neem_stats['sample_count'] else "Neem Statistics:  Not present in test folder")

    # Classification report
    all_eval_classes = sorted(list(set(y_true_names) | set(y_pred_names)))
    rep = classification_report(y_true_names, y_pred_names, labels=all_eval_classes, output_dict=True, zero_division=0)

    # Save Confusion Matrix
    cm = confusion_matrix(y_true_names, y_pred_names, labels=all_eval_classes)
    plt.figure(figsize=(max(8, len(all_eval_classes) * 0.4), max(6, len(all_eval_classes) * 0.35)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=all_eval_classes, yticklabels=all_eval_classes)
    plt.title("FloraSense AI - Vision Transformer Confusion Matrix (Real Data)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix plot to '{cm_path}'.")

    final_report = {
        "model": "Dual Vision Transformer Ensemble",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_test_samples": total,
        "top1_accuracy_pct": round(top1_acc * 100, 2),
        "top3_accuracy_pct": round(top3_acc * 100, 2),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
        "average_latency_ms": round(avg_latency, 2),
        "tulsi_metrics": tulsi_stats,
        "neem_metrics": neem_stats,
        "classification_report": rep
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"Saved evaluation report to '{output_report}'.")

    return final_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloraSense AI Real ViT Evaluation")
    parser.add_argument("--test_dir", type=str, default="./data/test", help="Directory of real test images")
    parser.add_argument("--output", type=str, default="evaluation_vit_report.json", help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(test_dir=args.test_dir, output_report=args.output)
