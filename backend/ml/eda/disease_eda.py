"""
Crop Pest and Disease Image Dataset - Exploratory Data Analysis (EDA)
KrishiMitra AI Agricultural Advisory Platform
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import random

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CROP_COLORS = {
    "Cashew": "#4CAF50",
    "Cassava": "#FF9800",
    "Maize": "#FFC107",
    "Tomato": "#E91E63",
    "Other": "#9C27B0"
}


def get_crop_family(class_name: str) -> str:
    """Extract crop family name from class name."""
    lower = class_name.lower()
    if "cashew" in lower:
        return "Cashew"
    elif "cassava" in lower:
        return "Cassava"
    elif "maize" in lower:
        return "Maize"
    elif "tomato" in lower:
        return "Tomato"
    return "Other"


def is_healthy(class_name: str) -> bool:
    """Check if class represents healthy plant foliage."""
    return "healthy" in class_name.lower()


def run_disease_eda(
    dataset_dir: Path,
    output_dir: Path,
    sample_limit_per_class: int = 50
) -> Dict[str, Any]:
    """
    Executes comprehensive EDA on crop disease leaf image dataset.
    Generates high-resolution visualizations and analytical descriptions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("[1/3] EXECUTING EDA ON CROP DISEASE IMAGE DATASET")
    print("=" * 70)
    print(f"Dataset Location: {dataset_dir.resolve()}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # 1. Scan classes and collect image paths
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {dataset_dir}")

    class_counts: Dict[str, int] = {}
    class_samples: Dict[str, List[Path]] = {}
    crop_counts: Dict[str, int] = {}
    healthy_count = 0
    diseased_count = 0

    widths: List[int] = []
    heights: List[int] = []
    aspect_ratios: List[float] = []

    rgb_healthy: List[np.ndarray] = []
    rgb_diseased: List[np.ndarray] = []

    random.seed(42)

    for c_dir in class_dirs:
        c_name = c_dir.name
        img_paths = [p for p in c_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        count = len(img_paths)
        class_counts[c_name] = count

        crop_fam = get_crop_family(c_name)
        crop_counts[crop_fam] = crop_counts.get(crop_fam, 0) + count

        if is_healthy(c_name):
            healthy_count += count
        else:
            diseased_count += count

        # Sample for image size and RGB stats
        sampled = random.sample(img_paths, min(sample_limit_per_class, count)) if count > 0 else []
        class_samples[c_name] = sampled

        for img_path in sampled:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h != 0 else 1.0)

                    img_rgb = img.convert("RGB")
                    # Sample average RGB
                    arr = np.array(img_rgb, dtype=np.float32)
                    mean_color = arr.mean(axis=(0, 1))  # [R, G, B]
                    if is_healthy(c_name):
                        rgb_healthy.append(mean_color)
                    else:
                        rgb_diseased.append(mean_color)
            except Exception:
                continue

    total_images = sum(class_counts.values())
    total_classes = len(class_counts)
    print(f"Total Identified Images: {total_images:,}")
    print(f"Total Disease/Pest Classes: {total_classes}")
    print(f"Crop Families Represented: {', '.join(crop_counts.keys())}")

    graph_descriptions = {}

    # =========================================================================
    # GRAPH 1: Class Distribution
    # =========================================================================
    fig_path1 = figures_dir / "01_disease_class_distribution.png"
    plt.figure(figsize=(14, 8), dpi=300)
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]
    bar_colors = [CROP_COLORS.get(get_crop_family(lbl), "#2196F3") for lbl in labels]

    bars = plt.barh(labels[::-1], counts[::-1], color=bar_colors[::-1], edgecolor="#333333", alpha=0.88)
    plt.title("KrishiMitra AI: Crop Disease Dataset Class Distribution", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Number of Labeled Leaf Images", fontsize=12, labelpad=10)
    plt.ylabel("Crop Disease / Condition Class", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                 va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_path1)
    plt.close()
    graph_descriptions["01_disease_class_distribution"] = (
        "Horizontal bar chart showcasing the total sample count across all 22 disease classes. "
        "Highlights balanced vs. imbalanced target categories, validating class weighting in cross-entropy loss."
    )

    # =========================================================================
    # GRAPH 2: Crop-Wise Breakdown (Donut Chart)
    # =========================================================================
    fig_path2 = figures_dir / "02_disease_crop_breakdown.png"
    plt.figure(figsize=(9, 8), dpi=300)
    sorted_crops = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)
    c_labels = [f"{k}\n({v:,} imgs)" for k, v in sorted_crops]
    c_vals = [v for _, v in sorted_crops]
    c_colors = [CROP_COLORS.get(k, "#999999") for k, _ in sorted_crops]

    wedges, texts, autotexts = plt.pie(
        c_vals, labels=c_labels, colors=c_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75,
        textprops=dict(color="#222222", fontsize=11, fontweight="bold"),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontsize(12)
    plt.title("Image Distribution by Crop Category", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(fig_path2)
    plt.close()
    graph_descriptions["02_disease_crop_breakdown"] = (
        "Donut chart summarizing the proportion of leaf images belonging to each crop family "
        "(Cashew, Cassava, Maize, Tomato). Illustrates multi-crop vision model generalizability."
    )

    # =========================================================================
    # GRAPH 3: Healthy vs. Diseased Ratio
    # =========================================================================
    fig_path3 = figures_dir / "03_disease_healthy_vs_diseased.png"
    plt.figure(figsize=(8, 7), dpi=300)
    status_labels = [f"Infected / Diseased\n({diseased_count:,} imgs)", f"Healthy Foliage\n({healthy_count:,} imgs)"]
    status_vals = [diseased_count, healthy_count]
    status_colors = ["#E53935", "#43A047"]

    plt.pie(
        status_vals, labels=status_labels, colors=status_colors, autopct="%1.1f%%",
        explode=[0.05, 0], startangle=70, pctdistance=0.6,
        textprops=dict(color="#111111", fontsize=12, fontweight="bold"),
        wedgeprops=dict(edgecolor="white", linewidth=2.5)
    )
    plt.title("Proportion of Healthy vs. Infected Foliage", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(fig_path3)
    plt.close()
    graph_descriptions["03_disease_healthy_vs_diseased"] = (
        "Pie chart contrasting healthy control leaves against pathological or pest-damaged specimens, "
        "ensuring the neural network learns positive disease markers rather than purely healthy plant morphology."
    )

    # =========================================================================
    # GRAPH 4: Image Resolution & Aspect Ratio Analysis
    # =========================================================================
    fig_path4 = figures_dir / "04_disease_image_resolutions.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot A: Width vs Height
    ax1.scatter(widths, heights, color="#1E88E5", alpha=0.5, edgecolors="none", s=25)
    ax1.plot([0, max(max(widths, default=500), max(heights, default=500))],
             [0, max(max(widths, default=500), max(heights, default=500))],
             linestyle="--", color="#FF5722", label="Square (1:1 Aspect)")
    ax1.set_title("Raw Image Dimension Scatter (Width vs. Height)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Width (pixels)", fontsize=11)
    ax1.set_ylabel("Height (pixels)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Subplot B: Aspect Ratio Distribution
    ax2.hist(aspect_ratios, bins=25, color="#00897B", edgecolor="#004D40", alpha=0.8)
    ax2.axvline(1.0, color="#FF5722", linestyle="--", linewidth=2, label="Aspect = 1.0 (Square)")
    ax2.set_title("Aspect Ratio Distribution (Width / Height)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Aspect Ratio", fontsize=11)
    ax2.set_ylabel("Sample Frequency", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.suptitle("Raw Image Spatial Geometry & Aspect Ratio Consistency", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path4, bbox_inches="tight")
    plt.close()
    graph_descriptions["04_disease_image_resolutions"] = (
        "Dual scatter-and-histogram analysis of original image pixel dimensions and aspect ratios. "
        "Confirms spatial uniformity, demonstrating whether pre-training bicubic resize (224x224) induces distortion."
    )

    # =========================================================================
    # GRAPH 5: RGB Color Channel Intensity Distribution
    # =========================================================================
    fig_path5 = figures_dir / "05_disease_rgb_channel_profiles.png"
    plt.figure(figsize=(10, 6), dpi=300)

    if rgb_healthy and rgb_diseased:
        h_arr = np.array(rgb_healthy)
        d_arr = np.array(rgb_diseased)

        channels = ["Red Channel", "Green Channel", "Blue Channel"]
        h_means = h_arr.mean(axis=0)
        d_means = d_arr.mean(axis=0)
        h_stds = h_arr.std(axis=0)
        d_stds = d_arr.std(axis=0)

        x = np.arange(len(channels))
        width = 0.35

        plt.bar(x - width/2, h_means, width, yerr=h_stds, capsize=6,
                label="Healthy Leaves", color="#4CAF50", edgecolor="#2E7D32", alpha=0.85)
        plt.bar(x + width/2, d_means, width, yerr=d_stds, capsize=6,
                label="Diseased / Pest Affected", color="#F44336", edgecolor="#C62828", alpha=0.85)

        plt.title("Average RGB Channel Intensity: Healthy vs. Diseased Foliage", fontsize=15, fontweight="bold", pad=15)
        plt.ylabel("Mean Pixel Intensity (0 - 255)", fontsize=12)
        plt.xticks(x, channels, fontsize=11, fontweight="bold")
        plt.ylim(0, 255)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(fig_path5)
    plt.close()
    graph_descriptions["05_disease_rgb_channel_profiles"] = (
        "Grouped bar chart displaying comparative RGB chromatic profiles between healthy and diseased leaf sets. "
        "Quantifies necrotic and chlorotic pigment shifts (elevated Red and decreased Green reflectance in diseased tissue)."
    )

    # =========================================================================
    # GRAPH 6: Visual Inspection Montage (1 Sample per Class)
    # =========================================================================
    fig_path6 = figures_dir / "06_disease_sample_montage.png"
    n_cols = 6
    n_rows = int(np.ceil(len(class_dirs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows), dpi=250)
    axes = axes.flatten()

    for idx, c_dir in enumerate(class_dirs):
        c_name = c_dir.name
        samples = class_samples.get(c_name, [])
        ax = axes[idx]

        if samples:
            img_path = samples[0]
            try:
                with Image.open(img_path) as im:
                    ax.imshow(im)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "No Images", ha="center", va="center")

        # Color border green if healthy, red if diseased
        border_color = "#2E7D32" if is_healthy(c_name) else "#C62828"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        # Truncate long class names for title
        short_name = c_name.replace("Cashew ", "Cashew\n").replace("Cassava ", "Cassava\n").replace("Maize ", "Maize\n").replace("Tomato ", "Tomato\n")
        ax.set_title(short_name, fontsize=8.5, fontweight="bold", pad=4)
        ax.axis("off")

    # Turn off unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("KrishiMitra AI: Representative Leaf Morphologies Across All 22 Diagnostic Classes",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(fig_path6, bbox_inches="tight")
    plt.close()
    graph_descriptions["06_disease_sample_montage"] = (
        "Photographic collage featuring an exemplar leaf image from every diagnostic class. "
        "Allows direct visual auditing of symptom severity, leaf pathology patterns, and lighting variations across classes."
    )

    print("[OK] Disease Dataset EDA completed. 6 figures generated successfully!")
    return {
        "dataset": "Crop Disease Image Dataset",
        "total_images": total_images,
        "total_classes": total_classes,
        "crop_breakdown": crop_counts,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "graphs": graph_descriptions
    }
