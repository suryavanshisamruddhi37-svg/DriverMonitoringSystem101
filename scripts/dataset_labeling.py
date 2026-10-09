"""
Manual visual labeling tool for the Driver Monitoring System.

You look at one image at a time and press a key to assign a label.
Nothing is classified automatically. There is no ML, YOLO, MediaPipe,
or face detection in this script.

Original files in Images/ are only read and copied.
Images/ and processed_images/ are never modified, renamed, moved, or deleted.
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Image types to include (checked by file extension, case-insensitive).
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Keys 1-6 map to these folder names under dataset/.
LABELS = {
    "1": "front",
    "2": "eyes_closed",
    "3": "left_diagonal",
    "4": "right_diagonal",
    "5": "left_side",
    "6": "right_side",
}

# CSV column names, in this order.
CSV_COLUMNS = ["filename", "source_folder", "label"]

# Window title shown by OpenCV.
WINDOW_NAME = "Driver Monitoring System - Manual Labeling"

# Largest width/height used only for on-screen display.
# The copied file in dataset/ stays at the original size.
MAX_DISPLAY_WIDTH = 960
MAX_DISPLAY_HEIGHT = 720


def project_paths():
    """Return the project root, source folders, dataset folder, and CSV path."""
    # This file lives in scripts/, so the project root is one level up.
    project_root = Path(__file__).resolve().parent.parent
    images_root = project_root / "Images"
    source_folders = [
        images_root / "front",
        images_root / "side angles",
    ]
    dataset_root = project_root / "dataset"
    csv_path = dataset_root / "labels.csv"
    return project_root, source_folders, dataset_root, csv_path


def collect_image_files(folder):
    """Return a sorted list of image files inside one folder (not nested)."""
    if not folder.is_dir():
        return []

    files = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def create_dataset_folders(dataset_root):
    """Create dataset/<label>/ folders if they do not exist."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    for label in LABELS.values():
        (dataset_root / label).mkdir(parents=True, exist_ok=True)


def load_already_labeled(csv_path):
    """
    Read labels.csv and return a set of already labeled images.

    Each item is (filename, source_folder). Using both values means the
    same file name in Images/front and Images/side angles can be labeled
    separately if needed.
    """
    labeled = set()
    if not csv_path.is_file():
        return labeled

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            source_folder = (row.get("source_folder") or "").strip()
            if filename and source_folder:
                labeled.add((filename, source_folder))
    return labeled


def ensure_csv_header(csv_path):
    """Create labels.csv with a header row if the file does not exist yet."""
    if csv_path.is_file():
        return

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_csv_row(csv_path, filename, source_folder, label):
    """Add one labeled image to labels.csv (append, do not rewrite the file)."""
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writerow(
            {
                "filename": filename,
                "source_folder": source_folder,
                "label": label,
            }
        )


def load_bgr_image(path):
    """
    Open an image for display and convert it to OpenCV BGR.

    Pillow is used first because some camera files are MPO/JPEG-based.
    convert("RGB") keeps the primary photo. This is read-only.
    """
    with Image.open(path) as img:
        rgb_image = img.convert("RGB")
        rgb_array = np.array(rgb_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def fit_for_display(bgr_image):
    """
    Shrink a large image so it fits on screen.

    This does not change the original file or the copy saved in dataset/.
    """
    height, width = bgr_image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return bgr_image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(bgr_image, new_size, interpolation=cv2.INTER_AREA)


def add_help_text(bgr_image, source_folder, filename, remaining, total):
    """Draw keyboard instructions on a copy of the display image."""
    display = bgr_image.copy()
    lines = [
        f"{source_folder} / {filename}   ({remaining} left of {total})",
        "1 front  2 eyes_closed  3 left_diagonal  4 right_diagonal",
        "5 left_side  6 right_side  N skip  Q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return display


def unique_destination(dest_path):
    """
    If dest_path already exists, add _1, _2, ... before the extension.

    This avoids overwriting a previous copy when two source folders share
    the same file name.
    """
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_into_dataset(source_path, dest_folder):
    """
    Copy the original image into dataset/<label>/.

    shutil.copy2 copies the file. It does not move or delete the original.
    """
    dest_path = unique_destination(dest_folder / source_path.name)
    shutil.copy2(source_path, dest_path)
    return dest_path


def key_to_action(key_code):
    """
    Convert an OpenCV waitKey code into a label, skip, quit, or ignore.

    OpenCV returns an integer. We keep only the last 8 bits, then compare
    it to the character you pressed. Uppercase and lowercase N/Q both work.
    """
    if key_code < 0:
        return None

    char = chr(key_code & 0xFF)

    if char in LABELS:
        return ("label", LABELS[char])
    if char in ("n", "N"):
        return ("skip", None)
    if char in ("q", "Q"):
        return ("quit", None)
    return None


def print_instructions():
    """Print the keyboard map in the terminal."""
    print("Keyboard:")
    print("  1 = front")
    print("  2 = eyes_closed")
    print("  3 = left_diagonal")
    print("  4 = right_diagonal")
    print("  5 = left_side")
    print("  6 = right_side")
    print("  N = skip (leave unlabeled for now)")
    print("  Q = quit")
    print()
    print("The original image is copied into dataset/<label>/. It is never moved.")
    print("Click the image window first so it receives keyboard input.")
    print()


def main():
    project_root, source_folders, dataset_root, csv_path = project_paths()

    print("=" * 60)
    print("Driver Monitoring System - Manual Visual Labeling")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("Original Images/ folder: read-only (copy only)")
    print(f"Labeled copies: {dataset_root}")
    print(f"CSV log: {csv_path}")
    print()

    create_dataset_folders(dataset_root)
    ensure_csv_header(csv_path)
    already_labeled = load_already_labeled(csv_path)

    # Build the list of images that still need a label.
    pending = []
    total_found = 0
    for folder in source_folders:
        image_files = collect_image_files(folder)
        total_found += len(image_files)
        source_folder_name = folder.name
        for path in image_files:
            if (path.name, source_folder_name) not in already_labeled:
                pending.append((path, source_folder_name))

    print(f"Images found: {total_found}")
    print(f"Already labeled (skipped): {len(already_labeled)}")
    print(f"Remaining to label: {len(pending)}")
    print()
    print_instructions()

    if not pending:
        print("Nothing left to label. All scanned images are already in labels.csv.")
        return

    labeled_now = 0
    skipped_now = 0
    failed_now = 0
    total_pending = len(pending)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for index, (path, source_folder_name) in enumerate(pending, start=1):
            remaining = total_pending - index + 1
            try:
                bgr_image = load_bgr_image(path)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                failed_now += 1
                print(f"Could not open {source_folder_name}/{path.name}: {error}")
                continue

            display = fit_for_display(bgr_image)
            display = add_help_text(
                display,
                source_folder_name,
                path.name,
                remaining,
                total_pending,
            )
            cv2.imshow(WINDOW_NAME, display)

            # Wait until the user presses a useful key.
            action = None
            while action is None:
                key_code = cv2.waitKey(0)
                action = key_to_action(key_code)

            kind, label = action
            if kind == "quit":
                print("Quit requested. Progress already saved in labels.csv.")
                break

            if kind == "skip":
                skipped_now += 1
                print(f"Skipped {source_folder_name}/{path.name}")
                continue

            # kind == "label"
            dest_folder = dataset_root / label
            dest_path = copy_into_dataset(path, dest_folder)
            append_csv_row(csv_path, path.name, source_folder_name, label)
            already_labeled.add((path.name, source_folder_name))
            labeled_now += 1
            print(
                f"Labeled {source_folder_name}/{path.name} -> "
                f"{label}/ ({dest_path.name})"
            )
    finally:
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("Session summary")
    print("=" * 60)
    print(f"Labeled this session: {labeled_now}")
    print(f"Skipped this session: {skipped_now}")
    print(f"Failed to open: {failed_now}")
    print(f"CSV file: {csv_path}")
    print()
    print("Finished. Original images in Images/ were not modified.")


if __name__ == "__main__":
    main()
