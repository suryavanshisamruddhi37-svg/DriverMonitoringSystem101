"""
Basic image operations and pixel manipulation for the Driver Monitoring System.

This script READS original images from Images/front and Images/side angles
and WRITES processed copies under processed_images/.

Original files in Images/ are never modified, renamed, moved, or deleted.
Rotation and horizontal flip are demonstrations only. They are not training
samples and they do not change class labels (front vs side angles).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Image types to include (checked by file extension, case-insensitive).
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Final square size used after the centered crop.
RESIZE_SIZE = 224

# Small controlled rotation used only as a demonstration.
ROTATION_DEGREES = 10

# JPEG quality for processed copies (originals are not overwritten).
JPEG_QUALITY = 95

# Subfolders for each operation, so results can be inspected separately.
# Class folders inside each of these stay exactly "front" and "side angles".
OPERATION_FOLDERS = [
    "jpeg_normalized",
    "cropped",
    "resized_224",
    "rotated_10",
    "flipped_horizontal",
    "rgb",
    "grayscale",
    "hsv_data",      # actual OpenCV HSV arrays (.npy), not viewable photos
    "hsv_preview",   # visual Hue / Saturation / Value panels for inspection
]


def project_paths():
    """Return the project root, source class folders, and processed output root."""
    # This file lives in scripts/, so the project root is one level up.
    project_root = Path(__file__).resolve().parent.parent
    images_root = project_root / "Images"
    class_folders = [
        images_root / "front",
        images_root / "side angles",
    ]
    output_root = project_root / "processed_images"
    return project_root, class_folders, output_root


def collect_image_files(folder):
    """Return a sorted list of image files inside one class folder (not nested)."""
    if not folder.is_dir():
        return []

    files = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def create_output_folders(output_root, class_names):
    """Create processed_images/<operation>/<class>/ folders if they do not exist."""
    for operation in OPERATION_FOLDERS:
        for class_name in class_names:
            folder = output_root / operation / class_name
            folder.mkdir(parents=True, exist_ok=True)


def load_rgb_image(path):
    """
    Open an image with Pillow and convert it to standard RGB.

    Pillow is used here because some front images are MPO/JPEG-based camera
    files. convert("RGB") keeps the primary photo and drops extra MPO frames.
    This is read-only: the file on disk is not changed.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


def center_crop_square(pil_image):
    """
    Crop a square from the center of a Pillow image.

    The crop side length is the smaller of width and height, so the largest
    centered square is kept. This is done before resizing to 224x224.
    """
    width, height = pil_image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return pil_image.crop((left, top, right, bottom))


def pil_to_bgr(pil_image):
    """Convert a Pillow RGB image to an OpenCV BGR NumPy array."""
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def save_jpeg(path, bgr_or_gray_image):
    """Save an OpenCV BGR or grayscale image as a JPEG file."""
    cv2.imwrite(
        str(path),
        bgr_or_gray_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )


def save_pil_jpeg(path, pil_image):
    """Save a Pillow RGB image as a standard JPEG file."""
    pil_image.save(path, format="JPEG", quality=JPEG_QUALITY)


def save_rgb_jpeg(path, bgr_image):
    """
    Convert OpenCV BGR to RGB, then save with Pillow.

    OpenCV's cv2.imwrite() expects BGR. Saving RGB data with OpenCV would
    swap the red and blue channels, so Pillow is used for the RGB folder.
    """
    rgb_array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_array).save(path, format="JPEG", quality=JPEG_QUALITY)


def rotate_image(bgr_image, degrees):
    """Rotate an OpenCV image around its center. Output size stays the same."""
    height, width = bgr_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(bgr_image, matrix, (width, height), borderValue=(0, 0, 0))


def make_hsv_preview(hsv_image):
    """
    Build a visual inspection image from an HSV array.

    OpenCV HSV is not a photo: channel 0 is Hue (0-179), channel 1 is
    Saturation (0-255), and channel 2 is Value (0-255). Saving that array
    as a JPEG would make software treat it as BGR/RGB, which is incorrect.

    This preview shows three panels side by side:
      - Hue, colored with a colormap so hue changes are easy to see
      - Saturation, as grayscale (brighter = more saturated)
      - Value, as grayscale (brighter = brighter)
    """
    hue, saturation, value = cv2.split(hsv_image)

    # OpenCV hue is 0-179. Scale to 0-255 so the colormap uses the full range.
    hue_scaled = (hue.astype(np.uint16) * 255 // 179).astype(np.uint8)
    hue_panel = cv2.applyColorMap(hue_scaled, cv2.COLORMAP_HSV)
    saturation_panel = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
    value_panel = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

    preview = np.hstack([hue_panel, saturation_panel, value_panel])

    # Small labels so the three panels are not mistaken for an RGB photo.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview, "Hue", (8, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, "Saturation", (224 + 8, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, "Value", (448 + 8, 18), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def process_one_image(source_path, class_name, output_root):
    """
    Run the requested operations on one image and save each result separately.

    Visual results use the original file stem plus .jpg.
    HSV arrays use the original file stem plus .npy.
    Class folders stay "front" or "side angles".
    """
    jpeg_name = source_path.stem + ".jpg"
    npy_name = source_path.stem + ".npy"

    # 1) Normalize to a standard RGB JPEG (MPO files become normal JPEGs).
    rgb_image = load_rgb_image(source_path)
    save_pil_jpeg(output_root / "jpeg_normalized" / class_name / jpeg_name, rgb_image)

    # 2) Centered square crop, saved so this step can be inspected on its own.
    cropped = center_crop_square(rgb_image)
    cropped_bgr = pil_to_bgr(cropped)
    save_jpeg(output_root / "cropped" / class_name / jpeg_name, cropped_bgr)

    # 3) Resize the cropped image to 224x224.
    resized = cv2.resize(
        cropped_bgr,
        (RESIZE_SIZE, RESIZE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    save_jpeg(output_root / "resized_224" / class_name / jpeg_name, resized)

    # The remaining operations are demonstrations on the 224x224 image.
    # 4) Small controlled rotation (+10 degrees). Demonstration only.
    rotated = rotate_image(resized, ROTATION_DEGREES)
    save_jpeg(output_root / "rotated_10" / class_name / jpeg_name, rotated)

    # 5) Horizontal flip. Demonstration only.
    #    The class folder is unchanged: a flipped front image stays in front/.
    #    These files are not extra training samples.
    flipped = cv2.flip(resized, 1)
    save_jpeg(output_root / "flipped_horizontal" / class_name / jpeg_name, flipped)

    # 6) Explicit RGB color space (OpenCV BGR -> RGB), saved with Pillow.
    save_rgb_jpeg(output_root / "rgb" / class_name / jpeg_name, resized)

    # 7) Grayscale (single channel).
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    save_jpeg(output_root / "grayscale" / class_name / jpeg_name, gray)

    # 8) HSV color space (OpenCV BGR -> HSV).
    #    hsv_data: the real HSV array for later processing (do not open as a photo).
    #    hsv_preview: a labeled Hue/Saturation/Value image for visual inspection.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    np.save(output_root / "hsv_data" / class_name / npy_name, hsv)
    hsv_preview = make_hsv_preview(hsv)
    save_jpeg(output_root / "hsv_preview" / class_name / jpeg_name, hsv_preview)


def main():
    project_root, class_folders, output_root = project_paths()
    class_names = [folder.name for folder in class_folders]

    print("=" * 60)
    print("Driver Monitoring System - Basic Image Operations")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("Original Images/ folder: read-only")
    print(f"Processed output: {output_root}")
    print()

    create_output_folders(output_root, class_names)

    total_found = 0
    total_processed = 0
    total_failed = 0
    failed_files = []
    processed_by_class = {}

    for folder in class_folders:
        class_name = folder.name
        print("-" * 60)
        print(f"Class folder: Images/{class_name}")

        if not folder.is_dir():
            print("  Status: folder not found")
            processed_by_class[class_name] = 0
            print()
            continue

        image_files = collect_image_files(folder)
        total_found += len(image_files)
        print(f"  Images found: {len(image_files)}")

        processed = 0
        for path in image_files:
            try:
                process_one_image(path, class_name, output_root)
                processed += 1
            except (UnidentifiedImageError, OSError, ValueError) as error:
                total_failed += 1
                failed_files.append((path, str(error)))
                print(f"  Skipped {path.name}: {error}")

        total_processed += processed
        processed_by_class[class_name] = processed
        print(f"  Processed: {processed}")
        print()

    print("=" * 60)
    print("Processing summary")
    print("=" * 60)
    print(f"Images found: {total_found}")
    print(f"Images processed: {total_processed}")
    print(f"Failed: {total_failed}")
    for class_name, count in processed_by_class.items():
        print(f"  {class_name}: {count}")

    print()
    print("Each processed result was saved in these folders:")
    for operation in OPERATION_FOLDERS:
        print(f"  {output_root / operation}")
        for class_name in class_names:
            print(f"    {class_name}/")

    print()
    print("Pipeline per image:")
    print("  1. Convert to standard RGB JPEG (normalizes MPO files)")
    print("  2. Center square crop")
    print("  3. Resize cropped image to 224x224")
    print("  4. Rotate 224x224 image by +10 degrees (demo only)")
    print("  5. Horizontal flip of 224x224 image (demo only, labels unchanged)")
    print("  6. Save explicit RGB image")
    print("  7. Convert 224x224 image to grayscale")
    print("  8. Convert BGR to HSV; save .npy data and a visual HSV preview")
    print()
    print("Notes:")
    print("  - hsv_data/*.npy is the real HSV array (H, S, V). Do not treat it as RGB.")
    print("  - hsv_preview/*.jpg is only for viewing Hue, Saturation, and Value.")
    print("  - flipped_horizontal/ is a demonstration, not extra training data.")
    print("  - Class folders remain front/ and side angles/.")

    if failed_files:
        print()
        print("Failed files:")
        for path, error in failed_files:
            print(f"  - {path.relative_to(project_root)}: {error}")

    print()
    print("Finished. Original images in Images/ were not modified.")


if __name__ == "__main__":
    main()
