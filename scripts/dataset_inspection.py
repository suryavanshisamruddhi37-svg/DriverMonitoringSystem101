"""
Dataset inspection for the Driver Monitoring System.

This script only reads images under Images/front and Images/side angles.
It does not modify, rename, move, delete, resize, crop, rotate, flip,
or convert any original files.
"""

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Image types to include (checked by file extension, case-insensitive).
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_paths():
    """Return the project root and the two image folders."""
    # This file lives in scripts/, so the project root is one level up.
    project_root = Path(__file__).resolve().parent.parent
    images_root = project_root / "Images"
    folders = [
        images_root / "front",
        images_root / "side angles",
    ]
    return project_root, folders


def collect_image_files(folder):
    """Return a sorted list of image files inside one folder (not nested)."""
    if not folder.is_dir():
        return []

    files = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def inspect_image(path):
    """
    Try to open an image and read its size and format.

    load() is used so truncated or corrupted files are detected.
    The file is only read; nothing is written back.
    """
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        # Pillow format names are like JPEG, PNG, BMP, WEBP.
        image_format = img.format or "UNKNOWN"
        return width, height, image_format


def print_counter_block(title, counter):
    """Print a Counter as a simple list, sorted by count then name."""
    print(title)
    if not counter:
        print("  (none)")
        return
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {key}: {count}")


def main():
    project_root, folders = project_paths()

    print("=" * 60)
    print("Driver Monitoring System - Dataset Inspection")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("Mode: read-only (original images are not changed)")
    print()

    total_files = 0
    total_readable = 0
    total_unreadable = 0
    all_extensions = Counter()
    all_formats = Counter()
    all_resolutions = Counter()
    all_unreadable = []

    for folder in folders:
        print("-" * 60)
        print(f"Folder: {folder.relative_to(project_root)}")

        if not folder.is_dir():
            print("  Status: folder not found")
            print()
            continue

        image_files = collect_image_files(folder)
        folder_count = len(image_files)
        total_files += folder_count
        print(f"  Image files found: {folder_count}")

        readable = 0
        unreadable = []
        extensions = Counter()
        formats = Counter()
        resolutions = Counter()

        for path in image_files:
            extension = path.suffix.upper().lstrip(".")
            extensions[extension] += 1
            try:
                width, height, image_format = inspect_image(path)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                unreadable.append((path, str(error)))
                continue

            readable += 1
            formats[image_format] += 1
            resolutions[(width, height)] += 1

        total_readable += readable
        total_unreadable += len(unreadable)
        all_extensions.update(extensions)
        all_formats.update(formats)
        all_resolutions.update(resolutions)
        all_unreadable.extend(unreadable)

        print(f"  Readable: {readable}")
        print(f"  Corrupted / unreadable: {len(unreadable)}")
        print_counter_block("  File extensions:", extensions)
        print_counter_block("  Detected formats (Pillow):", formats)
        print_counter_block("  Resolutions (width x height):", {
            f"{w} x {h}": count for (w, h), count in resolutions.items()
        })

        if unreadable:
            print("  Unreadable files:")
            for path, error in unreadable:
                print(f"    - {path.name}: {error}")
        print()

    print("=" * 60)
    print("Overall summary")
    print("=" * 60)
    print(f"Total images: {total_files}")
    print(f"Readable: {total_readable}")
    print(f"Corrupted / unreadable: {total_unreadable}")
    print_counter_block("File extensions:", all_extensions)
    print_counter_block("Detected formats (Pillow):", all_formats)
    if "MPO" in all_formats:
        print("Note: MPO is a JPEG-based camera format. The files are still valid images.")
    print_counter_block("Resolutions (width x height):", {
        f"{w} x {h}": count for (w, h), count in all_resolutions.items()
    })

    if all_unreadable:
        print("All unreadable files:")
        for path, error in all_unreadable:
            print(f"  - {path.relative_to(project_root)}: {error}")
    else:
        print("All scanned images opened successfully.")

    print()
    print("Inspection finished. No original images were modified.")


if __name__ == "__main__":
    main()
