from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main\backend\ml\dataset\processed"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

print("=" * 70)
print("CHECKING PROCESSED DATASET")
print("=" * 70)

corrupted = []
total = 0

for split in ["train", "val", "test"]:
    split_dir = DATASET_DIR / split

    print(f"\nChecking: {split}")

    for image_path in split_dir.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

        except Exception as e:
            corrupted.append((image_path, str(e)))
            print(f"CORRUPTED: {image_path}")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Total images checked : {total}")
print(f"Corrupted images     : {len(corrupted)}")

if corrupted:
    print("\nCorrupted files:")
    for path, error in corrupted:
        print(f"\n{path}")
        print(f"Error: {error}")
else:
    print("\nAll images are valid!")

print("\n" + "=" * 70)