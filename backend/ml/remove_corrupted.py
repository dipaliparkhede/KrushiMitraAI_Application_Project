from pathlib import Path
from PIL import Image, ImageFile

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

removed = 0
checked = 0

print("=" * 70)
print("DEEP IMAGE VALIDATION")
print("=" * 70)

for split in ["train", "val", "test"]:

    split_dir = DATASET_DIR / split

    print(f"\nChecking {split}...")

    for image_path in split_dir.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        checked += 1

        try:
            # Open image
            with Image.open(image_path) as image:

                # Force PIL to actually read all image data
                image.load()

                # Convert to RGB to ensure conversion works
                image.convert("RGB")

        except Exception as e:

            print(f"\nCORRUPTED:")
            print(image_path)
            print(f"Reason: {e}")

            try:
                image_path.unlink()
                removed += 1
            except Exception as delete_error:
                print(f"Could not delete: {delete_error}")


print("\n" + "=" * 70)
print("DEEP VALIDATION COMPLETE")
print("=" * 70)

print(f"Images checked : {checked}")
print(f"Images removed : {removed}")

print("=" * 70)