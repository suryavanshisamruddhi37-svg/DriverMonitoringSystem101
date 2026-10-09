from pathlib import Path
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "ear_dataset" / "images"
LABEL_DIR = PROJECT_ROOT / "ear_dataset" / "labels"

# Temporary folders prevent filename collisions
TEMP_IMAGE_DIR = PROJECT_ROOT / "ear_dataset" / "_temp_images"
TEMP_LABEL_DIR = PROJECT_ROOT / "ear_dataset" / "_temp_labels"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FIND MATCHED PAIRS
# ============================================================

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

pairs = []

for image in sorted(images):
    label = LABEL_DIR / f"{image.stem}.txt"

    if label.exists():
        pairs.append((image, label))


# ============================================================
# DISPLAY VERIFICATION
# ============================================================

print("=" * 60)
print("EAR DATASET SAFE RENAMING")
print("=" * 60)

print(f"Images found       : {len(images)}")
print(f"Matched annotations: {len(pairs)}")
print(f"Unmatched images   : {len(images) - len(pairs)}")

print()

if not pairs:
    print("No image-label pairs found.")
    print("Nothing was renamed.")
    exit()


print("The following matched pairs will be renamed:")
print()

for index, (image, label) in enumerate(pairs, start=1):
    print(
        f"{index:03d}: "
        f"{image.name}  <->  {label.name}"
    )

print()
print("=" * 60)

confirmation = input(
    "Type YES to rename these matched pairs: "
).strip()

if confirmation != "YES":
    print("Cancelled. No files were changed.")
    exit()


# ============================================================
# CREATE TEMPORARY FOLDERS
# ============================================================

TEMP_IMAGE_DIR.mkdir(exist_ok=True)
TEMP_LABEL_DIR.mkdir(exist_ok=True)


# ============================================================
# FIRST MOVE TO TEMPORARY NAMES
# ============================================================

for index, (image, label) in enumerate(pairs, start=1):

    temp_image = TEMP_IMAGE_DIR / f"temp_{index:04d}{image.suffix.lower()}"
    temp_label = TEMP_LABEL_DIR / f"temp_{index:04d}.txt"

    shutil.move(str(image), str(temp_image))
    shutil.move(str(label), str(temp_label))


# ============================================================
# MOVE TO FINAL NAMES
# ============================================================

for index, (image, label) in enumerate(pairs, start=1):

    temp_image = TEMP_IMAGE_DIR / f"temp_{index:04d}{image.suffix.lower()}"
    temp_label = TEMP_LABEL_DIR / f"temp_{index:04d}.txt"

    final_image = IMAGE_DIR / f"ear_{index:03d}{image.suffix.lower()}"
    final_label = LABEL_DIR / f"ear_{index:03d}.txt"

    shutil.move(str(temp_image), str(final_image))
    shutil.move(str(temp_label), str(final_label))


# ============================================================
# REMOVE TEMPORARY FOLDERS
# ============================================================

if TEMP_IMAGE_DIR.exists():
    TEMP_IMAGE_DIR.rmdir()

if TEMP_LABEL_DIR.exists():
    TEMP_LABEL_DIR.rmdir()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("RENAMING COMPLETED")
print("=" * 60)

print(f"Images renamed : {len(pairs)}")
print(f"Labels renamed : {len(pairs)}")

print()
print("Example:")

if pairs:
    first_ext = pairs[0][0].suffix.lower()

    print(f"ear_001{first_ext}")
    print("ear_001.txt")

print()
print("Image and YOLO label filenames remain matched.")