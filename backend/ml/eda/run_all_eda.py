"""
KrishiMitra AI - Master Exploratory Data Analysis (EDA) Pipeline
Runs comprehensive EDA across:
1. Crop Disease Image Dataset
2. Crop Recommendation Tabular Dataset
3. Crop Yield Prediction Tabular Dataset

Produces high-resolution figures and a structured Markdown summary report.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adjust path to enable standalone script execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from eda.disease_eda import run_disease_eda
from eda.crop_eda import run_crop_eda
from eda.yield_eda import run_yield_eda


def generate_eda_markdown_report(
    disease_results: dict,
    crop_results: dict,
    yield_results: dict,
    output_dir: Path
) -> Path:
    """Generates a comprehensive Markdown documentation report of all EDA findings."""
    report_file = output_dir / "EDA_REPORT.md"

    lines = []
    lines.append("# 🌾 KrishiMitra AI — Exploratory Data Analysis (EDA) Report")
    lines.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append("This document provides an exhaustive exploratory data analysis (EDA) across the three core machine learning datasets powering the KrishiMitra AI platform:\n")
    lines.append("1. **Crop Pest & Disease Leaf Image Dataset** (Computer Vision — EfficientNet-B0)")
    lines.append("2. **Crop Recommendation Soil & Climate Dataset** (Multi-Class Classification — Random Forest)")
    lines.append("3. **Crop Yield Agricultural Productivity Dataset** (Regression — Random Forest Regressor)\n")
    lines.append("---\n")

    # ---------------------------------------------------------
    # SECTION 1: Crop Disease Dataset
    # ---------------------------------------------------------
    lines.append("## 🌿 1. Crop Pest & Disease Leaf Image Dataset\n")
    lines.append(f"- **Total Identified Images**: {disease_results['total_images']:,}")
    lines.append(f"- **Total Diagnostic Classes**: {disease_results['total_classes']}")
    lines.append(f"- **Healthy Samples**: {disease_results['healthy_count']:,} ({disease_results['healthy_count'] / disease_results['total_images'] * 100:.1f}%)")
    lines.append(f"- **Infected/Diseased Samples**: {disease_results['diseased_count']:,} ({disease_results['diseased_count'] / disease_results['total_images'] * 100:.1f}%)")
    lines.append(f"- **Crop Families**: {', '.join(disease_results['crop_breakdown'].keys())}\n")
    lines.append("### Diagnostic Visualizations & Descriptions:\n")

    for graph_name, desc in disease_results["graphs"].items():
        lines.append(f"#### 📊 `{graph_name}.png`")
        lines.append(f"![{graph_name}](figures/{graph_name}.png)\n")
        lines.append(f"**Short Description & Agricultural Insight:**  \n{desc}\n")

    lines.append("---\n")

    # ---------------------------------------------------------
    # SECTION 2: Crop Recommendation Dataset
    # ---------------------------------------------------------
    lines.append("## 🌾 2. Crop Recommendation Dataset (Soil & Climate)\n")
    lines.append(f"- **Total Agricultural Observations**: {crop_results['total_records']:,}")
    lines.append(f"- **Unique Crop Varieties**: {crop_results['unique_crops']}")
    lines.append(f"- **Nutrient & Climatic Features**: {', '.join(crop_results['features'])}\n")
    lines.append("### Agronomic Visualizations & Descriptions:\n")

    for graph_name, desc in crop_results["graphs"].items():
        lines.append(f"#### 📊 `{graph_name}.png`")
        lines.append(f"![{graph_name}](figures/{graph_name}.png)\n")
        lines.append(f"**Short Description & Agricultural Insight:**  \n{desc}\n")

    lines.append("---\n")

    # ---------------------------------------------------------
    # SECTION 3: Crop Yield Prediction Dataset
    # ---------------------------------------------------------
    lines.append("## 📈 3. Crop Yield Prediction Dataset (Macro Productivity)\n")
    lines.append(f"- **Total Historical Records**: {yield_results['total_records']:,}")
    lines.append(f"- **Covered Crops**: {yield_results['unique_crops']}")
    lines.append(f"- **Covered Indian States & UTs**: {yield_results['unique_states']}")
    lines.append(f"- **Macro Features**: {', '.join(yield_results['features'])}\n")
    lines.append("### Productivity Visualizations & Descriptions:\n")

    for graph_name, desc in yield_results["graphs"].items():
        lines.append(f"#### 📊 `{graph_name}.png`")
        lines.append(f"![{graph_name}](figures/{graph_name}.png)\n")
        lines.append(f"**Short Description & Agricultural Insight:**  \n{desc}\n")

    lines.append("---\n")
    lines.append("### Summary of Artifacts Generated\n")
    lines.append(f"- All high-resolution PNG figures (300 DPI) are saved in: `{output_dir / 'figures'}`\n")
    lines.append("- Full Markdown documentation report saved in: `EDA_REPORT.md`\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_file


def main():
    parser = argparse.ArgumentParser(description="KrishiMitra AI - Master EDA Runner")
    parser.add_argument(
        "--disease-dir",
        type=Path,
        default=Path("backend/ml/dataset/crop disease dataset"),
        help="Path to raw crop disease image dataset folder"
    )
    parser.add_argument(
        "--crop-path",
        type=Path,
        default=None,
        help="Path to Crop_recommendation.csv"
    )
    parser.add_argument(
        "--yield-path",
        type=Path,
        default=None,
        help="Path to crop_yield.csv or .parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/ml/eda/outputs"),
        help="Directory to save figures and reports"
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("KRISHIMITRA AI - COMPREHENSIVE MULTI-DATASET EDA PIPELINE")
    print("=" * 70)

    # 1. Disease EDA
    disease_results = run_disease_eda(
        dataset_dir=args.disease_dir,
        output_dir=output_dir
    )

    # 2. Crop Recommendation EDA
    crop_results = run_crop_eda(
        dataset_path=args.crop_path,
        output_dir=output_dir
    )

    # 3. Crop Yield EDA
    yield_results = run_yield_eda(
        dataset_path=args.yield_path,
        output_dir=output_dir
    )

    # 4. Generate Markdown Report
    report_file = generate_eda_markdown_report(
        disease_results=disease_results,
        crop_results=crop_results,
        yield_results=yield_results,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] COMPREHENSIVE EDA COMPLETED!")
    print("=" * 70)
    print(f"High-Resolution Figures: {(output_dir / 'figures').resolve()}")
    print(f"Full Documentation Report: {report_file.resolve()}\n")


if __name__ == "__main__":
    main()
