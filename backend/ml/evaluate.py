from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main"
)

TEST_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "dataset"
    / "processed"
    / "test"
)

MODEL_DIR = PROJECT_ROOT / "backend" / "model"

MODEL_PATH = MODEL_DIR / "best_model.pth"

CLASSES_PATH = MODEL_DIR / "classes.json"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CROP DISEASE MODEL EVALUATION")
print("=" * 70)

print("\nTest dataset:")
print(TEST_DIR)

print("\nModel:")
print(MODEL_PATH)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD CLASSES
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

print("\nNumber of classes:", len(classes))


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("\nTest images:", len(test_dataset))

# Make sure class ordering matches
if test_dataset.classes != classes:
    print("\nWARNING: Class ordering mismatch!")

    print("\nModel classes:")
    print(classes)

    print("\nTest classes:")
    print(test_dataset.classes)

    raise ValueError("Class ordering mismatch")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(classes)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# best_model.pth contains a state_dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []
all_labels = []

correct = 0
total = 0

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(f"\nTest Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1 Score       : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

cm_path = MODEL_DIR / "confusion_matrix.npy"

np.save(
    cm_path,
    cm
)

print("\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = MODEL_DIR / "evaluation_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("CROP DISEASE MODEL EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Test images: {len(test_dataset)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(report)


print("\nEvaluation report saved:")
print(report_path)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)