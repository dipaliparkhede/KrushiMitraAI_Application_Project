from pathlib import Path
import random
import shutil


# ============================================================
# DATASET PATH
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main\backend\ml\dataset\crop disease dataset"
)


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = Path(
    r"C:\Users\Dipali\OneDrive\Desktop\KrishiMitraAI_project-main\backend\ml\dataset\processed"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}"
    )


# ============================================================
# REMOVE OLD PROCESSED DATA
# ============================================================

if OUTPUT_DIR.exists():

    print("\nRemoving old processed dataset...")

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    (
        OUTPUT_DIR / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DATASET PREPARATION")
print("=" * 70)

print("\nSource:")
print(DATASET_DIR)

print("\nDestination:")
print(OUTPUT_DIR)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_images = 0


for class_folder in sorted(
    DATASET_DIR.iterdir()
):

    if not class_folder.is_dir():
        continue


    class_name = (
        class_folder.name
        .strip()
        .replace(" ", "_")
    )


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    images = [

        image

        for image in class_folder.rglob("*")

        if image.suffix.lower()
        in IMAGE_EXTENSIONS

    ]


    if not images:

        print(
            f"\nWARNING: No images found in "
            f"{class_folder.name}"
        )

        continue


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(images)


    # --------------------------------------------------------
    # CALCULATE SPLIT
    # --------------------------------------------------------

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = int(
        total *
        (TRAIN_RATIO + VAL_RATIO)
    )


    train_images = images[
        :train_end
    ]

    val_images = images[
        train_end:val_end
    ]

    test_images = images[
        val_end:
    ]


    # --------------------------------------------------------
    # CREATE SPLITS
    # --------------------------------------------------------

    splits = {

        "train": train_images,

        "val": val_images,

        "test": test_images

    }


    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    for split, split_images in splits.items():

        destination_folder = (
            OUTPUT_DIR
            / split
            / class_name
        )


        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        for image_path in split_images:

            destination = (
                destination_folder
                / image_path.name
            )


            shutil.copy2(
                image_path,
                destination
            )


    total_images += total


    print(
        f"{class_name:<40}"
        f"Total: {total:>5} | "
        f"Train: {len(train_images):>5} | "
        f"Val: {len(val_images):>5} | "
        f"Test: {len(test_images):>5}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal images processed: {total_images}"
)

print(
    f"\nProcessed dataset:"
)

print(OUTPUT_DIR)