"""
Crop Recommendation Model Comparison & Confusion Matrix Benchmark
KrishiMitra-AI: Benchmarking 6 Machine Learning Models on 440 Test Samples (22 Crop Classes)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "dataset" / "Crop_recommendation.csv"


def benchmark_crop_models():
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Random Forest (Production)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tp_total = int(np.trace(cm))
        total_samples = int(np.sum(cm))
        errors = total_samples - tp_total

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

        results.append({
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp_total,
            "errors": errors,
            "total": total_samples,
            "cm": cm
        })

    # Sort by accuracy descending
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    print("=" * 105)
    print("KRISHIMITRA-AI: CROP RECOMMENDATION MODEL BENCHMARK (CONFUSION MATRIX METRICS)")
    print("=" * 105)
    print(f"Dataset: Crop_recommendation.csv | Test Samples: {len(X_test)} | Crop Classes: {len(np.unique(y))}")
    print("=" * 105)

    header = (
        f"| {'Rank':<4} | {'Model Name':<28} | {'Accuracy':<9} | {'Precision':<9} "
        f"| {'Recall':<9} | {'F1-Score':<9} | {'TP / 440':<9} | {'Errors':<7} | {'Error Rate':<10} |"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        err_rate = (r['errors'] / r['total']) * 100
        print(
            f"| {i:<4} | {r['name']:<28} | {r['accuracy']:>8.2f}% | {r['precision']:>8.2f}% "
            f"| {r['recall']:>8.2f}% | {r['f1']:>8.2f}% | {r['tp']:>9} | {r['errors']:>7} | {err_rate:>9.2f}% |"
        )
    print(sep)
    return results


if __name__ == "__main__":
    benchmark_crop_models()
