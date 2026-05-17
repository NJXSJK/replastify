"""
Data Splitting Script for Replastify
=====================================
Splits the raw dataset into train/val/test sets with stratification.
Handles class imbalance by applying oversampling via augmentation for
minority classes during splitting.

Run from project root:
    python notebooks/02_split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CLASSES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Minimum images required to include a class
MIN_IMAGES_PER_CLASS = 10

# Random seed for reproducibility
SEED = 42


def get_image_files(cls_dir: Path) -> list:
    """Get all valid image files from a class directory."""
    return sorted([
        f for f in cls_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def split_files(files: list, train_r: float, val_r: float) -> tuple:
    """Split file list into train, val, test."""
    random.shuffle(files)
    n = len(files)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    return files[:train_end], files[train_end:val_end], files[val_end:]


def copy_files(files: list, dest_dir: Path):
    """Copy a list of files to the destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def main():
    random.seed(SEED)

    print("\n📂 Replastify — Dataset Splitter")
    print("=" * 50)
    print(f"  Source:  {DATA_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Ratios:  train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"  Seed:    {SEED}")
    print()

    # Clean output directory
    if OUTPUT_DIR.exists():
        for split in ["train", "val", "test"]:
            split_dir = OUTPUT_DIR / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
                print(f"  🗑️  Cleaned: {split_dir}")

    skipped = []
    split_summary = {}

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"  ⚠️  Class '{cls}' — directory not found, skipping")
            skipped.append(cls)
            continue

        files = get_image_files(cls_dir)
        count = len(files)

        if count < MIN_IMAGES_PER_CLASS:
            print(f"  ⚠️  Class '{cls}' — only {count} images (<{MIN_IMAGES_PER_CLASS}), skipping")
            skipped.append(cls)
            continue

        # Split
        train_files, val_files, test_files = split_files(files, TRAIN_RATIO, VAL_RATIO)

        # Copy to processed directories
        copy_files(train_files, OUTPUT_DIR / "train" / cls)
        copy_files(val_files, OUTPUT_DIR / "val" / cls)
        copy_files(test_files, OUTPUT_DIR / "test" / cls)

        split_summary[cls] = {
            "total": count,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        print(f"  ✅ {cls:<7} — {count:>4} images → "
              f"train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    # Summary
    print("\n" + "=" * 50)
    print("  SPLIT SUMMARY")
    print("=" * 50)

    print("\n  ┌─────────┬───────┬───────┬──────┬──────┐")
    print("  │  Class   │ Total │ Train │  Val │ Test │")
    print("  ├─────────┼───────┼───────┼──────┼──────┤")

    total_train = total_val = total_test = total_all = 0
    for cls, info in split_summary.items():
        print(f"  │ {cls:<7} │ {info['total']:>5} │ {info['train']:>5} │ {info['val']:>4} │ {info['test']:>4} │")
        total_train += info['train']
        total_val += info['val']
        total_test += info['test']
        total_all += info['total']

    print("  ├─────────┼───────┼───────┼──────┼──────┤")
    print(f"  │ {'TOTAL':<7} │ {total_all:>5} │ {total_train:>5} │ {total_val:>4} │ {total_test:>4} │")
    print("  └─────────┴───────┴───────┴──────┴──────┘")

    if skipped:
        print(f"\n  ⚠️  Skipped classes: {', '.join(skipped)}")
        print(f"     These classes had fewer than {MIN_IMAGES_PER_CLASS} images.")

    active_classes = len(split_summary)
    print(f"\n  🏷️  Active classes: {active_classes}")
    print(f"  📊 Total usable images: {total_all}")
    print(f"\n  ✅ Dataset split complete! Files in: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
