
from pathlib import Path
from collections import defaultdict
import hashlib

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Store images by their SHA-256 hash
hash_groups = defaultdict(list)
class_counts = {}

def calculate_hash(image_path):
    """Calculate a unique content hash for an image file."""
    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    print("=" * 65)
    print("DRIVER MONITORING SYSTEM: DATASET AUDIT")
    print("=" * 65)

    total_images = 0

    # Scan each class folder
    for class_folder in sorted(DATASET_ROOT.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        images = [
            path for path in class_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[class_name] = len(images)
        total_images += len(images)

        for image_path in images:
            image_hash = calculate_hash(image_path)
            hash_groups[image_hash].append(
                (class_name, image_path)
            )

    print(f"\nTotal source images: {total_images}")

    print("\nImage count by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:<20} {count:>4}")

    duplicate_groups = [
        entries for entries in hash_groups.values()
        if len(entries) > 1
    ]

    conflicting_groups = [
        entries for entries in duplicate_groups
        if len({class_name for class_name, _ in entries}) > 1
    ]

    print(f"\nExact duplicate groups: {len(duplicate_groups)}")
    print(f"Groups with conflicting labels: {len(conflicting_groups)}")

    for index, entries in enumerate(duplicate_groups, start=1):
        print(f"\nDuplicate group {index}:")

        for class_name, image_path in entries:
            print(f"  [{class_name}] {image_path.relative_to(PROJECT_ROOT)}")

        labels = {class_name for class_name, _ in entries}
        if len(labels) > 1:
            print("  WARNING: SAME IMAGE HAS DIFFERENT CLASS LABELS")

    print("\nAudit complete. No files were changed.")


if __name__ == "__main__":
    main()
