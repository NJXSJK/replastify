"""
Data Exploration Script for Replastify
======================================
Analyzes the raw plastic dataset: class distribution, sample images, 
image dimensions, and quality checks.

Run from project root:
    python notebooks/01_data_exploration.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images_per_class(data_dir: Path) -> dict:
    """Count the number of images in each class folder."""
    counts = {}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            files = [f for f in cls_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
            counts[cls] = len(files)
        else:
            counts[cls] = 0
    return counts


def get_image_dimensions(data_dir: Path, sample_size: int = 50) -> dict:
    """Sample images from each class and record their dimensions."""
    dimensions = {}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        files = [f for f in cls_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        sampled = files[:sample_size]
        dims = []
        for f in sampled:
            try:
                with Image.open(f) as img:
                    dims.append(img.size)  # (width, height)
            except Exception:
                continue
        dimensions[cls] = dims
    return dimensions


def plot_class_distribution(counts: dict, output_path: Path):
    """Bar chart of images per class."""
    classes = list(counts.keys())
    values = list(counts.values())
    total = sum(values)

    colors = ['#2d6a4f', '#40916c', '#52b788', '#74c69d', '#95d5b2', '#b7e4c7', '#d8f3dc']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, values, color=colors, edgecolor='white', linewidth=1.5)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlabel('Plastic Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
    ax.set_title(f'Dataset Class Distribution (Total: {total} images)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_dimension_distribution(dimensions: dict, output_path: Path):
    """Scatter plot of image widths vs heights across classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2d6a4f', '#40916c', '#52b788', '#74c69d', '#95d5b2', '#b7e4c7', '#d8f3dc']

    for i, cls in enumerate(CLASSES):
        if cls not in dimensions or not dimensions[cls]:
            continue
        widths = [d[0] for d in dimensions[cls]]
        heights = [d[1] for d in dimensions[cls]]
        ax.scatter(widths, heights, label=cls, alpha=0.6, s=40, color=colors[i])

    ax.set_xlabel('Width (px)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Height (px)', fontsize=13, fontweight='bold')
    ax.set_title('Image Dimension Distribution (sampled)', fontsize=15, fontweight='bold', pad=15)
    ax.legend(title='Class', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def plot_sample_images(data_dir: Path, output_path: Path, samples_per_class: int = 3):
    """Grid of sample images from each class."""
    fig, axes = plt.subplots(len(CLASSES), samples_per_class, 
                              figsize=(4 * samples_per_class, 3.5 * len(CLASSES)))

    for row, cls in enumerate(CLASSES):
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            for col in range(samples_per_class):
                axes[row, col].axis('off')
                axes[row, col].set_title(f'{cls} — no data', fontsize=10)
            continue

        files = sorted([f for f in cls_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
        
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(files):
                try:
                    img = Image.open(files[col]).convert('RGB')
                    ax.imshow(img)
                    w, h = img.size
                    ax.set_title(f'{cls} ({w}×{h})', fontsize=10, fontweight='bold')
                except Exception:
                    ax.set_title(f'{cls} — error', fontsize=10)
            else:
                ax.set_title(f'{cls} — N/A', fontsize=10)
            ax.axis('off')

    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")


def print_summary(counts: dict, dimensions: dict):
    """Print a text summary of the dataset."""
    total = sum(counts.values())
    print("\n" + "=" * 60)
    print("  REPLASTIFY — Dataset Summary")
    print("=" * 60)
    print(f"\n  📁 Dataset location: {DATA_DIR}")
    print(f"  📊 Total images: {total}")
    print(f"  🏷️  Number of classes: {len(CLASSES)}")
    print()

    print("  ┌─────────┬────────┬──────────────┐")
    print("  │  Class   │ Images │  % of Total  │")
    print("  ├─────────┼────────┼──────────────┤")
    for cls in CLASSES:
        count = counts.get(cls, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  │ {cls:<7} │ {count:>6} │ {pct:>5.1f}% {bar:<5} │")
    print("  └─────────┴────────┴──────────────┘")

    # Dimension summary
    all_widths = []
    all_heights = []
    for cls, dims in dimensions.items():
        all_widths.extend([d[0] for d in dims])
        all_heights.extend([d[1] for d in dims])

    if all_widths:
        print(f"\n  📐 Image dimensions (sampled):")
        print(f"     Width  — min: {min(all_widths)}px, max: {max(all_widths)}px, "
              f"mean: {np.mean(all_widths):.0f}px")
        print(f"     Height — min: {min(all_heights)}px, max: {max(all_heights)}px, "
              f"mean: {np.mean(all_heights):.0f}px")

    # Imbalance warnings
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        if max_count > 0 and min_count / max_count < 0.1:
            print(f"\n  ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            weak = [cls for cls, c in counts.items() if c < 20]
            if weak:
                print(f"     Classes with <20 images: {', '.join(weak)}")
                print(f"     Consider: dropping these classes or collecting more data")

    print("\n" + "=" * 60)


def main():
    print("\n🔍 Running Replastify Data Exploration...\n")

    if not DATA_DIR.exists():
        print(f"  ❌ Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # 1. Count images
    print("  📊 Counting images per class...")
    counts = count_images_per_class(DATA_DIR)

    # 2. Get dimensions
    print("  📐 Sampling image dimensions...")
    dimensions = get_image_dimensions(DATA_DIR)

    # 3. Print summary
    print_summary(counts, dimensions)

    # 4. Generate plots
    print("\n  📈 Generating visualizations...")
    plot_class_distribution(counts, OUTPUT_DIR / "class_distribution.png")
    plot_dimension_distribution(dimensions, OUTPUT_DIR / "dimension_distribution.png")
    plot_sample_images(DATA_DIR, OUTPUT_DIR / "sample_images.png")

    print(f"\n  ✅ All outputs saved to: {OUTPUT_DIR}/")
    print("  Done!\n")


if __name__ == "__main__":
    main()
