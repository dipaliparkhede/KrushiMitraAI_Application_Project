from pathlib import Path
import json
import copy

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "dataset"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "model"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("CROP DISEASE MODEL TRAINING")
print("=" * 70)

print("\nDataset:")
print(DATASET_DIR)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# HYPERPARAMETERS
# ============================================================

IMAGE_SIZE = 224

# RTX 2050 has 4GB VRAM
BATCH_SIZE = 16

NUM_WORKERS = 0

# Stage 1
CLASSIFIER_EPOCHS = 2

# Stage 2
FINETUNE_EPOCHS = 8


CLASSIFIER_LR = 0.001

FINETUNE_LR = 0.0001

WEIGHT_DECAY = 0.0001


# ============================================================
# CHECK DATASET
# ============================================================

for folder in [
    "train",
    "val",
    "test"
]:

    path = DATASET_DIR / folder

    if not path.exists():

        raise FileNotFoundError(
            f"\nDataset folder not found:\n{path}"
        )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")


train_dataset = datasets.ImageFolder(

    DATASET_DIR / "train",

    transform=train_transform
)


val_dataset = datasets.ImageFolder(

    DATASET_DIR / "val",

    transform=val_transform
)


test_dataset = datasets.ImageFolder(

    DATASET_DIR / "test",

    transform=val_transform
)


classes = train_dataset.classes

NUM_CLASSES = len(classes)


print(
    "\nNumber of classes:",
    NUM_CLASSES
)


# ============================================================
# PRINT CLASSES
# ============================================================

print("\nClasses:")

for index, class_name in enumerate(classes):

    print(
        f"{index:2d} -> {class_name}"
    )


# ============================================================
# VERIFY CLASSES
# ============================================================

if train_dataset.classes != val_dataset.classes:

    raise ValueError(
        "Train and validation classes are different."
    )


if train_dataset.classes != test_dataset.classes:

    raise ValueError(
        "Train and test classes are different."
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

classes_file = (
    MODEL_DIR
    / "classes.json"
)

with open(
    classes_file,
    "w"
) as file:

    json.dump(
        classes,
        file,
        indent=4
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


print("\nDataset sizes:")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

targets = np.array(
    train_dataset.targets
)


class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(
        NUM_CLASSES
    ),

    y=targets
)


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float32

).to(DEVICE)


criterion = nn.CrossEntropyLoss(

    weight=class_weights
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET
# ============================================================

print("\nLoading EfficientNet-B0...")


model = efficientnet_b0(

    weights=(
        EfficientNet_B0_Weights.DEFAULT
    )
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

in_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(

    in_features,

    NUM_CLASSES
)


model = model.to(DEVICE)


print(
    "Model loaded successfully."
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )


    return average_loss, accuracy


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    epochs,
    learning_rate,
    stage_name
):

    print("\n")
    print("=" * 70)
    print(stage_name)
    print("=" * 70)


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=WEIGHT_DECAY
    )


    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.3,

            patience=2
        )
    )


    best_accuracy = 0.0

    best_weights = copy.deepcopy(
        model.state_dict()
    )


    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        progress = tqdm(

            train_loader,

            desc=(
                f"Epoch "
                f"{epoch + 1}/{epochs}"
            )
        )


        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate()
        )


        scheduler.step(
            val_accuracy
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            best_weights = copy.deepcopy(
                model.state_dict()
            )


            model_path = (
                MODEL_DIR
                / "best_model.pth"
            )


            torch.save(

                {
                    "model_state_dict":
                        best_weights,

                    "classes":
                        classes,

                    "num_classes":
                        NUM_CLASSES,

                    "image_size":
                        IMAGE_SIZE

                },

                model_path
            )


            print(
                "\n✓ Best model saved:"
            )

            print(
                model_path
            )


    model.load_state_dict(
        best_weights
    )


    return best_accuracy


# ============================================================
# STAGE 1
# ============================================================

print("\nStarting Stage 1...")


for parameter in model.features.parameters():

    parameter.requires_grad = False


stage1_accuracy = train_model(

    epochs=CLASSIFIER_EPOCHS,

    learning_rate=CLASSIFIER_LR,

    stage_name=(
        "STAGE 1 - CLASSIFIER TRAINING"
    )
)


# ============================================================
# STAGE 2
# ============================================================

print("\nStarting Stage 2...")


for parameter in model.features.parameters():

    parameter.requires_grad = True


stage2_accuracy = train_model(

    epochs=FINETUNE_EPOCHS,

    learning_rate=FINETUNE_LR,

    stage_name=(
        "STAGE 2 - FULL FINE-TUNING"
    )
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "final_model.pth"
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "classes":
            classes,

        "num_classes":
            NUM_CLASSES,

        "image_size":
            IMAGE_SIZE

    },

    final_model_path
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nStage 1 Best Validation Accuracy:"
    f" {stage1_accuracy * 100:.2f}%"
)

print(
    f"Stage 2 Best Validation Accuracy:"
    f" {stage2_accuracy * 100:.2f}%"
)

print(
    "\nBest model:"
)

print(
    MODEL_DIR / "best_model.pth"
)

print(
    "\nFinal model:"
)

print(
    final_model_path
)

print(
    "\nClasses:"
)

print(
    classes_file
)