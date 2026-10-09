"""
Train / Validation / Test Dataset Split
Driver Monitoring System

This script reads the manually labeled dataset from:

dataset/
    front/
    eyes_closed/
    left_diagonal/
    right_diagonal/
    left_side/
    right_side/

and creates:

dataset_split/
    train/
    validation/
    test/

The original dataset/ folder is NEVER modified.
Images are copied, not moved.
"""

from pathlib import Path
import shutil
import random


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Fixed seed makes the split reproducible.
RANDOM_SEED = 42

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CLASSES = [
    "front",
    "eyes_closed",
    "left_diagonal",
    "right_diagonal",
    "left_side",
    "right_side",
]


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

OUTPUT_ROOT = PROJECT_ROOT / "dataset_split"


# ---------------------------------------------------------
# CREATE OUTPUT FOLDERS
# ---------------------------------------------------------

def create_output_folders():
    """
    Create train, validation and test folders
    for every class.
    """

    splits = ["train", "validation", "test"]

    for split in splits:
        for class_name in CLASSES:

            folder = OUTPUT_ROOT / split / class_name

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ---------------------------------------------------------
# COLLECT IMAGES
# ---------------------------------------------------------

def collect_images(folder):
    """
    Return all supported image files from one class folder.
    """

    images = []

    if not folder.is_dir():
        return images

    for path in folder.iterdir():

        if (
            path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        ):
            images.append(path)

    return images


# ---------------------------------------------------------
# COPY IMAGE
# ---------------------------------------------------------

def copy_image(source, destination_folder):
    """
    Copy image into destination.

    Original image is never moved or modified.
    """

    destination = destination_folder / source.name

    shutil.copy2(
        source,
        destination
    )


# ---------------------------------------------------------
# SPLIT ONE CLASS
# ---------------------------------------------------------

def split_class(class_name):
    """
    Split one class into:

    70% train
    15% validation
    15% test
    """

    source_folder = DATASET_ROOT / class_name

    images = collect_images(source_folder)

    # Shuffle images randomly.
    random.shuffle(images)

    total = len(images)

    # Calculate split positions.
    train_count = int(total * TRAIN_RATIO)

    validation_count = int(
        total * VALIDATION_RATIO
    )

    test_count = (
        total
        - train_count
        - validation_count
    )

    # Split image list.
    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # Copy training images.
    for image in train_images:

        copy_image(
            image,
            OUTPUT_ROOT
            / "train"
            / class_name
        )

    # Copy validation images.
    for image in validation_images:

        copy_image(
            image,
            OUTPUT_ROOT
            / "validation"
            / class_name
        )

    # Copy testing images.
    for image in test_images:

        copy_image(
            image,
            OUTPUT_ROOT
            / "test"
            / class_name
        )

    return (
        total,
        len(train_images),
        len(validation_images),
        len(test_images),
    )


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 65)

    print(
        "Driver Monitoring System "
        "- Dataset Split"
    )

    print("=" * 65)

    print(
        f"Source dataset: {DATASET_ROOT}"
    )

    print(
        f"Output dataset: {OUTPUT_ROOT}"
    )

    print()

    print(
        "Split ratio:"
    )

    print(
        "Train      : 70%"
    )

    print(
        "Validation : 15%"
    )

    print(
        "Test       : 15%"
    )

    print()

    print(
        "Original dataset/ will NOT "
        "be modified."
    )

    print("-" * 65)

    # Set random seed.
    random.seed(RANDOM_SEED)

    # Create folders.
    create_output_folders()

    total_all = 0
    train_all = 0
    validation_all = 0
    test_all = 0

    # Process every class.
    for class_name in CLASSES:

        (
            total,
            train_count,
            validation_count,
            test_count,
        ) = split_class(class_name)

        total_all += total
        train_all += train_count
        validation_all += validation_count
        test_all += test_count

        print()

        print(
            f"Class: {class_name}"
        )

        print(
            f"  Total      : {total}"
        )

        print(
            f"  Train      : {train_count}"
        )

        print(
            f"  Validation : {validation_count}"
        )

        print(
            f"  Test       : {test_count}"
        )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print()

    print("=" * 65)

    print(
        "Final Dataset Split Summary"
    )

    print("=" * 65)

    print(
        f"Original images : {total_all}"
    )

    print(
        f"Training images : {train_all}"
    )

    print(
        f"Validation images : "
        f"{validation_all}"
    )

    print(
        f"Testing images : {test_all}"
    )

    print()

    print(
        "Total copied images:",
        train_all
        + validation_all
        + test_all,
    )

    print()

    print(
        "Finished."
    )

    print(
        "Original dataset/ was "
        "not modified."
    )


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main()