from pathlib import Path
from PIL import Image


# ============================================================
# DATASET PATH
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main\backend\ml\dataset\crop disease dataset"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CROP PEST AND DISEASE DATASET ANALYSIS")
print("=" * 70)

print("\nDataset path:")
print(DATASET_DIR)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}"
    )


print("\nDataset found successfully!")


# ============================================================
# VARIABLES
# ============================================================

total_images = 0
total_classes = 0

corrupted_images = []


# ============================================================
# ANALYZE EACH CLASS
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for class_folder in sorted(
    DATASET_DIR.iterdir()
):

    # Ignore files
    if not class_folder.is_dir():
        continue


    total_classes += 1

    image_count = 0


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    for image_path in class_folder.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue


        image_count += 1
        total_images += 1


        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:

                image.verify()

        except Exception:

            corrupted_images.append(
                str(image_path)
            )


    print(
        f"{class_folder.name:<40}"
        f"{image_count:>6} images"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Total classes    : {total_classes}"
)

print(
    f"Total images     : {total_images}"
)

print(
    f"Corrupted images : {len(corrupted_images)}"
)


# ============================================================
# CORRUPTED FILES
# ============================================================

if corrupted_images:

    print("\n" + "=" * 70)
    print("CORRUPTED IMAGES")
    print("=" * 70)

    for image in corrupted_images:

        print(image)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)