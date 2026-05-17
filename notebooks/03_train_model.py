"""
Replastify — Model Training Pipeline
=====================================
Two-stage transfer learning for plastic classification.
Trains EfficientNet-B0 (primary) and ResNet50 (baseline).

Designed to run on Kaggle with GPU. Adjust DATA_DIR and OUTPUT_DIR as needed.

Usage:
    python 03_train_model.py
"""

import os
import sys
import json
import time
import copy
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — Change these paths for Kaggle
# ═══════════════════════════════════════════════════════════════════

# For Kaggle, change to: '/kaggle/input/replastify/data/processed'
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# For Kaggle, change to: '/kaggle/working/outputs'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "outputs"

# For Kaggle, change to: '/kaggle/working/models'
MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"

SEED = 42
NUM_CLASSES = 6
CLASS_NAMES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
# Note: torchvision.datasets.ImageFolder sorts class names alphabetically

# ImageNet normalization (required for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ═══════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def get_transforms(input_size=224):
    """Get train and val/test transforms."""
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(input_size + 32),  # 256 for 224 input
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tfm, val_tfm


def get_dataloaders(data_dir, train_tfm, val_tfm, batch_size=32):
    """Create data loaders with weighted sampling for class imbalance."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tfm)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tfm)

    # Print class mapping
    print(f"\n  Class mapping: {train_dataset.class_to_idx}")

    # Weighted sampler for class imbalance
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Class counts: {dict(zip(train_dataset.classes, class_counts))}")

    return train_loader, val_loader, train_dataset.classes


# ═══════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    """
    Build a model with a custom classification head.
    Returns model and the feature dimension of the backbone.
    """
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        backbone_name = "features"

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features  # 2048
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        backbone_name = "layer4"  # last residual block

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, backbone_name


def freeze_backbone(model, model_name):
    """Freeze all backbone parameters (Stage 1)."""
    if model_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False


def unfreeze_last_blocks(model, model_name):
    """Unfreeze last 2-3 blocks for fine-tuning (Stage 2)."""
    if model_name == "efficientnet_b0":
        # EfficientNet-B0 has features[0] through features[8]
        # Unfreeze features[6], features[7], features[8]
        for param in model.features[6:].parameters():
            param.requires_grad = True
        # Keep frozen BN layers in eval mode (prevent running stats corruption)
        for module in model.features[:6].modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    elif model_name == "resnet50":
        # Unfreeze layer4 (last residual block)
        for param in model.layer4.parameters():
            param.requires_grad = True
        # Keep earlier BN layers in eval mode
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and not name.startswith("layer4"):
                module.eval()


def count_parameters(model):
    """Count trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()

    # Keep frozen BN layers in eval (called every epoch because model.train() resets it)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, stage_name, model_save_path, patience=7):
    """
    Run a complete training stage with early stopping.
    Returns training history and best model state_dict.
    """
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    total_params, trainable_params = count_parameters(model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print()

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(f"  Epoch {epoch+1:>2}/{num_epochs} │ "
              f"Train: {train_acc:.4f} loss={train_loss:.4f} │ "
              f"Val: {val_acc:.4f} loss={val_loss:.4f} │ "
              f"LR: {current_lr:.2e} │ {elapsed:.1f}s", end="")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)
            epochs_no_improve = 0
            print(" ✅ best")
        else:
            epochs_no_improve += 1
            print(f" ({epochs_no_improve}/{patience})")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_training_curves(history_s1, history_s2, model_name, output_dir):
    """Plot training curves for both stages."""
    # Combine histories
    train_loss = history_s1["train_loss"] + history_s2["train_loss"]
    val_loss = history_s1["val_loss"] + history_s2["val_loss"]
    train_acc = history_s1["train_acc"] + history_s2["train_acc"]
    val_acc = history_s1["val_acc"] + history_s2["val_acc"]
    lrs = history_s1["lr"] + history_s2["lr"]
    stage1_epochs = len(history_s1["train_loss"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(train_loss, label='Train Loss', color='#2d6a4f', linewidth=2)
    axes[0].plot(val_loss, label='Val Loss', color='#e63946', linewidth=2)
    axes[0].axvline(x=stage1_epochs - 0.5, color='gray', linestyle='--', alpha=0.5, label='Stage 2 start')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(train_acc, label='Train Acc', color='#2d6a4f', linewidth=2)
    axes[1].plot(val_acc, label='Val Acc', color='#e63946', linewidth=2)
    axes[1].axvline(x=stage1_epochs - 0.5, color='gray', linestyle='--', alpha=0.5, label='Stage 2 start')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Learning Rate
    axes[2].plot(lrs, color='#457b9d', linewidth=2)
    axes[2].axvline(x=stage1_epochs - 0.5, color='gray', linestyle='--', alpha=0.5, label='Stage 2 start')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title(f'{model_name} — LR Schedule')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    save_path = output_dir / f"training_curves_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def train_model(model_name, data_dir, output_dir, model_dir, device):
    """Full two-stage training pipeline for one model."""

    print(f"\n{'#'*60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'#'*60}")

    # Setup
    train_tfm, val_tfm = get_transforms(input_size=224)
    train_loader, val_loader, class_names = get_dataloaders(data_dir, train_tfm, val_tfm, batch_size=32)

    # Build model
    model, backbone_name = build_model(model_name, num_classes=NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model_save_path = model_dir / f"best_{model_name}.pth"

    # ── Stage 1: Feature Extraction (Frozen Backbone) ──
    freeze_backbone(model, model_name)

    # Only pass trainable params to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_s1 = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-2)
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=10)

    history_s1 = train_stage(
        model, train_loader, val_loader, criterion,
        optimizer_s1, scheduler_s1, device,
        num_epochs=10,
        stage_name=f"Stage 1: Feature Extraction ({model_name})",
        model_save_path=model_save_path,
        patience=10,  # Don't early-stop stage 1 (it's short)
    )

    # ── Stage 2: Fine-Tuning (Partial Unfreeze) ──
    unfreeze_last_blocks(model, model_name)

    # Differential learning rate: backbone=1e-4, head=1e-3
    if model_name == "efficientnet_b0":
        param_groups = [
            {"params": model.features[6:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    elif model_name == "resnet50":
        param_groups = [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ]

    optimizer_s2 = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s2, T_0=5, T_mult=1)

    history_s2 = train_stage(
        model, train_loader, val_loader, criterion,
        optimizer_s2, scheduler_s2, device,
        num_epochs=20,
        stage_name=f"Stage 2: Fine-Tuning ({model_name})",
        model_save_path=model_save_path,
        patience=7,
    )

    # Plot training curves
    plot_training_curves(history_s1, history_s2, model_name, output_dir)

    # Save combined history
    combined_history = {
        "stage1": history_s1,
        "stage2": history_s2,
        "model_name": model_name,
        "best_val_acc": max(history_s1["val_acc"] + history_s2["val_acc"]),
        "class_names": class_names,
    }

    history_path = output_dir / f"history_{model_name}.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"  💾 History saved: {history_path}")
    print(f"  💾 Best model saved: {model_save_path}")

    return combined_history


def main():
    set_seed(SEED)

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Verify data directory
    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        print("   Make sure you've run 02_split_dataset.py first!")
        sys.exit(1)

    start_time = time.time()

    # Train both models
    results = {}

    # 1. EfficientNet-B0 (Primary)
    results["efficientnet_b0"] = train_model("efficientnet_b0", DATA_DIR, OUTPUT_DIR, MODEL_DIR, device)

    # 2. ResNet50 (Baseline)
    results["resnet50"] = train_model("resnet50", DATA_DIR, OUTPUT_DIR, MODEL_DIR, device)

    # Summary comparison
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"{'='*60}")
    print(f"\n  ┌─────────────────┬───────────────┐")
    print(f"  │     Model       │ Best Val Acc  │")
    print(f"  ├─────────────────┼───────────────┤")
    for name, res in results.items():
        acc = res["best_val_acc"]
        marker = " 🏆" if acc == max(r["best_val_acc"] for r in results.values()) else ""
        print(f"  │ {name:<15} │ {acc:>10.4f}   │{marker}")
    print(f"  └─────────────────┴───────────────┘")
    print(f"\n  Models saved to: {MODEL_DIR}/")
    print(f"  Plots saved to:  {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
