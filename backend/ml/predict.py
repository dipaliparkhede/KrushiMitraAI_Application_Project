from pathlib import Path
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "model"
    / "best_model.pth"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "backend"
    / "model"
    / "classes.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD CLASSES
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    classes = json.load(f)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

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
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()

    confidence = confidence.item()

    predicted_class = classes[predicted_index]

    # Top 3 predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        top_predictions.append({
            "class": classes[index.item()],
            "confidence": probability.item() * 100
        })

    return {
        "prediction": predicted_class,
        "confidence": confidence * 100,
        "top_predictions": top_predictions
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CROP DISEASE IMAGE PREDICTION")
    print("=" * 70)

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    try:

        result = predict_image(image_path)

        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)

        print(
            f"\nPredicted class : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence      : "
            f"{result['confidence']:.2f}%"
        )

        print("\nTop 3 predictions:")

        for i, prediction in enumerate(
            result["top_predictions"],
            start=1
        ):

            print(
                f"{i}. "
                f"{prediction['class']:<35}"
                f"{prediction['confidence']:.2f}%"
            )

        print("\n" + "=" * 70)

    except Exception as e:

        print("\nPrediction failed:")
        print(e)