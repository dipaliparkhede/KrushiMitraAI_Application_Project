import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS CONFIGURATION
# ============================================================
# Resolve project root dynamically based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "backend" / "model"

CM_PATH = MODEL_DIR / "confusion_matrix.npy"
CLASSES_PATH = MODEL_DIR / "classes.json"

RAW_PLOT_PATH = MODEL_DIR / "confusion_matrix_raw.png"
NORM_PLOT_PATH = MODEL_DIR / "confusion_matrix_normalized.png"
LATEX_TABLE_PATH = MODEL_DIR / "research_paper_table.tex"


def main():
    print("=" * 70)
    print("KRISHIMITRA-AI: CROP DISEASE MODEL METRICS & CONFUSION MATRIX")
    print("=" * 70)

    # 1. Check if required files exist
    if not CM_PATH.exists():
        raise FileNotFoundError(f"Confusion matrix not found at: {CM_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes list not found at: {CLASSES_PATH}")

    # 2. Load classes and confusion matrix
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    cm = np.load(CM_PATH)
    num_classes = len(classes)
    total_samples = int(np.sum(cm))

    print(f"\nTotal Disease Classes : {num_classes}")
    print(f"Total Test Samples    : {total_samples}")

    # 3. Compute Metrics from Confusion Matrix
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    accuracy = np.sum(tp) / total_samples

    # Macro averages (unweighted)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Weighted averages (weighted by class support)
    weighted_precision = np.sum(precision * support) / total_samples
    weighted_recall = np.sum(recall * support) / total_samples
    weighted_f1 = np.sum(f1 * support) / total_samples

    # 4. Display Overall Summary
    print("\n" + "=" * 70)
    print("OVERALL MODEL PERFORMANCE SUMMARY (FOR RESEARCH PAPER)")
    print("=" * 70)
    print(f"Overall Accuracy           : {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f"Weighted Precision         : {weighted_precision * 100:.2f}% ({weighted_precision:.4f})")
    print(f"Weighted Recall            : {weighted_recall * 100:.2f}% ({weighted_recall:.4f})")
    print(f"Weighted F1-Score          : {weighted_f1 * 100:.2f}% ({weighted_f1:.4f})")
    print("-" * 70)
    print(f"Macro Average Precision    : {macro_precision * 100:.2f}% ({macro_precision:.4f})")
    print(f"Macro Average Recall       : {macro_recall * 100:.2f}% ({macro_recall:.4f})")
    print(f"Macro Average F1-Score     : {macro_f1 * 100:.2f}% ({macro_f1:.4f})")
    print("=" * 70)

    # 5. Display Class-Wise Metrics
    print("\nCLASS-WISE EVALUATION BREAKDOWN:")
    print(f"{'Class Name':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 68)
    for i, cls in enumerate(classes):
        print(f"{cls:<28} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {int(support[i]):<8}")

    # 6. Plot and Save Raw Confusion Matrix Heatmap
    display_labels = [c.replace("_", " ") for c in classes]

    plt.figure(figsize=(15, 13))
    sns.set_theme(font_scale=0.8)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Sample Count"}
    )
    plt.title("Confusion Matrix (Sample Counts) - Crop Disease Detection (EfficientNet-B0)", fontsize=13, pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("True Label", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RAW_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Raw Confusion Matrix plot: {RAW_PLOT_PATH}")

    # 7. Plot and Save Normalized Confusion Matrix Heatmap
    cm_norm = cm.astype(float) / support[:, np.newaxis]

    plt.figure(figsize=(15, 13))
    sns.set_theme(font_scale=0.8)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Normalized Recall Rate"}
    )
    plt.title("Normalized Confusion Matrix - Crop Disease Detection (EfficientNet-B0)", fontsize=13, pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("True Label", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(NORM_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Normalized Confusion Matrix plot: {NORM_PLOT_PATH}")

    # 8. Export LaTeX Table for Direct Paper Inclusion
    with open(LATEX_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write("% Copy and paste this table into your research paper LaTeX source\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Class-wise and Overall Performance Metrics of Crop Disease Model}\n")
        f.write("\\label{tab:crop_disease_metrics}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Class Name} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\ \\hline\n")
        for i, cls in enumerate(classes):
            latex_cls = cls.replace("_", "\\_")
            f.write(f"{latex_cls} & {precision[i]:.4f} & {recall[i]:.4f} & {f1[i]:.4f} & {int(support[i])} \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\textbf{{Macro Average}} & {macro_precision:.4f} & {macro_recall:.4f} & {macro_f1:.4f} & {total_samples} \\\\\n")
        f.write(f"\\textbf{{Weighted Average}} & \\textbf{{{weighted_precision:.4f}}} & \\textbf{{{weighted_recall:.4f}}} & \\textbf{{{weighted_f1:.4f}}} & {total_samples} \\\\\n")
        f.write(f"\\textbf{{Overall Accuracy}} & \\multicolumn{{4}}{{c|}}{{\\textbf{{{accuracy * 100:.2f}\\%}}}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[Saved] LaTeX Table for Paper: {LATEX_TABLE_PATH}")
    print("\n" + "=" * 70)
    print("ALL ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
