"""
Replastify — Model Evaluation Pipeline
=======================================
Generates comprehensive metrics, visualizations, and Grad-CAM heatmaps
for trained plastic classification models.

Usage:
    python 04_evaluate_model.py

Prerequisites:
    - Run 03_train_model.py first to generate model .pth files
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "outputs"

NUM_CLASSES = 6
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def build_model(model_name, num_classes=NUM_CLASSES):
    """Build model architecture (same as training script, no pretrained weights)."""
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    return model


def load_trained_model(model_name, model_dir, device):
    """Load a trained model from disk."""
    model = build_model(model_name)
    model_path = model_dir / f"best_{model_name}.pth"

    if not model_path.exists():
        print(f"  ❌ Model not found: {model_path}")
        return None

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  ✅ Loaded: {model_path}")
    return model


# ═══════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(model, loader, device):
    """Get all predictions, true labels, and probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_topk_accuracy(probs, labels, k=3):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i])
    return correct / len(labels)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(labels, preds, class_names, model_name, output_dir):
    """Plot normalized confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {model_name}\n(Normalized by True Labels)',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {path}")


def plot_per_class_f1(labels, preds, class_names, model_name, output_dir):
    """Horizontal bar chart of per-class F1 scores."""
    f1_per_class = f1_score(labels, preds, average=None)

    colors = ['#d62828' if f < 0.5 else '#f77f00' if f < 0.7 else '#2d6a4f' for f in f1_per_class]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(class_names, f1_per_class, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, f1_per_class):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold', fontsize=11)

    # Threshold lines
    ax.axvline(x=0.5, color='#d62828', linestyle='--', alpha=0.5, label='Fail threshold (0.50)')
    ax.axvline(x=0.7, color='#f77f00', linestyle='--', alpha=0.5, label='Good threshold (0.70)')

    ax.set_xlabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Class F1 Score — {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"per_class_f1_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {path}")


def plot_confidence_distribution(probs, labels, preds, model_name, output_dir):
    """Histogram of prediction confidence for correct vs incorrect."""
    max_probs = np.max(probs, axis=1)
    correct_mask = (preds == labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_probs[correct_mask], bins=30, alpha=0.7, label='Correct', color='#2d6a4f', edgecolor='white')
    ax.hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='#d62828', edgecolor='white')

    ax.set_xlabel('Max Prediction Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Confidence Distribution — {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"confidence_dist_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {path}")


def plot_misclassified(dataset, preds, labels, probs, class_names, model_name, output_dir, max_show=12):
    """Grid of misclassified images with true/predicted labels."""
    wrong_indices = np.where(preds != labels)[0]

    if len(wrong_indices) == 0:
        print("  🎉 No misclassifications!")
        return

    # Sort by confidence (most confident wrong predictions first)
    wrong_confs = [np.max(probs[i]) for i in wrong_indices]
    sorted_wrong = [wrong_indices[i] for i in np.argsort(wrong_confs)[::-1]]
    show_indices = sorted_wrong[:max_show]

    cols = 4
    rows = (len(show_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    # De-normalize for display
    inv_mean = np.array(IMAGENET_MEAN)
    inv_std = np.array(IMAGENET_STD)

    for idx, ax in enumerate(axes):
        if idx < len(show_indices):
            sample_idx = show_indices[idx]
            img_tensor, _ = dataset[sample_idx]
            img = img_tensor.numpy().transpose(1, 2, 0)
            img = img * inv_std + inv_mean
            img = np.clip(img, 0, 1)

            true_cls = class_names[labels[sample_idx]]
            pred_cls = class_names[preds[sample_idx]]
            conf = np.max(probs[sample_idx]) * 100

            ax.imshow(img)
            ax.set_title(f'True: {true_cls}\nPred: {pred_cls} ({conf:.0f}%)',
                        fontsize=9, color='red', fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'Most Confident Misclassifications — {model_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / f"misclassified_{model_name}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(model_name, model, test_loader, test_dataset, class_names, device, output_dir):
    """Full evaluation of one model."""

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'='*60}")

    # Get predictions
    preds, labels, probs = get_predictions(model, test_loader, device)

    # ── Core Metrics ──
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    kappa = cohen_kappa_score(labels, preds)
    top3_acc = compute_topk_accuracy(probs, labels, k=3)

    print(f"\n  ┌────────────────────┬──────────┐")
    print(f"  │       Metric       │  Value   │")
    print(f"  ├────────────────────┼──────────┤")
    print(f"  │ Accuracy           │ {acc:>7.4f}  │")
    print(f"  │ Macro F1           │ {f1_macro:>7.4f}  │")
    print(f"  │ Weighted F1        │ {f1_weighted:>7.4f}  │")
    print(f"  │ Cohen's Kappa      │ {kappa:>7.4f}  │")
    print(f"  │ Top-3 Accuracy     │ {top3_acc:>7.4f}  │")
    print(f"  └────────────────────┴──────────┘")

    # ── Classification Report ──
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(f"\n  Classification Report:\n{report}")

    # Save report to file
    report_path = output_dir / f"classification_report_{model_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n\n")
        f.write(report)
    print(f"  📄 Saved: {report_path}")

    # ── Visualizations ──
    plot_confusion_matrix(labels, preds, class_names, model_name, output_dir)
    plot_per_class_f1(labels, preds, class_names, model_name, output_dir)
    plot_confidence_distribution(probs, labels, preds, model_name, output_dir)
    plot_misclassified(test_dataset, preds, labels, probs, class_names, model_name, output_dir)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "kappa": float(kappa),
        "top3_accuracy": float(top3_acc),
        "per_class_f1": {cls: float(f) for cls, f in zip(class_names, f1_score(labels, preds, average=None))},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Test data loader
    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        print(f"  ❌ Test directory not found: {test_dir}")
        sys.exit(1)

    test_dataset = datasets.ImageFolder(test_dir, transform=test_tfm)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    class_names = test_dataset.classes
    print(f"  Test set: {len(test_dataset)} images, classes: {class_names}")

    # Evaluate each model
    all_results = {}
    for model_name in ["efficientnet_b0", "resnet50"]:
        model = load_trained_model(model_name, MODEL_DIR, device)
        if model is None:
            continue
        results = evaluate_model(model_name, model, test_loader, test_dataset, class_names, device, OUTPUT_DIR)
        all_results[model_name] = results

    if len(all_results) < 2:
        print("\n  ⚠️  Not all models available for comparison")
        if all_results:
            # Save whatever we have
            with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
        return

    # ── Model Comparison ──
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")

    print(f"\n  ┌────────────────────┬─────────────────┬─────────────────┐")
    print(f"  │       Metric       │ EfficientNet-B0 │    ResNet50     │")
    print(f"  ├────────────────────┼─────────────────┼─────────────────┤")

    metrics_to_compare = ["accuracy", "f1_macro", "f1_weighted", "kappa", "top3_accuracy"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1", "Cohen's Kappa", "Top-3 Accuracy"]

    best_model = None
    best_f1 = 0

    for label, key in zip(metric_labels, metrics_to_compare):
        v1 = all_results.get("efficientnet_b0", {}).get(key, 0)
        v2 = all_results.get("resnet50", {}).get(key, 0)
        m1 = " 🏆" if v1 > v2 else "   "
        m2 = " 🏆" if v2 > v1 else "   "
        print(f"  │ {label:<18} │  {v1:.4f}{m1}      │  {v2:.4f}{m2}      │")

    print(f"  └────────────────────┴─────────────────┴─────────────────┘")

    # Determine winner
    for name, res in all_results.items():
        if res["f1_macro"] > best_f1:
            best_f1 = res["f1_macro"]
            best_model = name

    print(f"\n  🏆 Best model: {best_model} (Macro F1 = {best_f1:.4f})")

    # Save all results
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  💾 Saved: {OUTPUT_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
