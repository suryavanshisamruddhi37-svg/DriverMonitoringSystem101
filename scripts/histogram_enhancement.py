from pathlib import Path
import cv2
import numpy as np


# ============================================================
# DRIVER MONITORING SYSTEM
# Image Enhancement using Histogram Techniques
# ============================================================

print("=" * 60)
print("Driver Monitoring System - Histogram Enhancement")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use images from the ear annotation dataset
EAR_IMAGE_DIR = PROJECT_ROOT / "ear_dataset" / "images"

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed_images"
    / "histogram_enhancement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FIND DRIVER IMAGES
# ============================================================

image_files = sorted(
    list(EAR_IMAGE_DIR.glob("*.jpg")) +
    list(EAR_IMAGE_DIR.glob("*.jpeg")) +
    list(EAR_IMAGE_DIR.glob("*.JPG")) +
    list(EAR_IMAGE_DIR.glob("*.JPEG")) +
    list(EAR_IMAGE_DIR.glob("*.png")) +
    list(EAR_IMAGE_DIR.glob("*.PNG"))
)


if not image_files:
    raise FileNotFoundError(
        f"No images found in:\n{EAR_IMAGE_DIR}"
    )


# Select the first available image
IMAGE_PATH = image_files[0]

print()
print(f"Selected driver image:")
print(IMAGE_PATH)
print()


# ============================================================
# 3. READ IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise ValueError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


print(
    f"Image size: "
    f"{image.shape[1]} x {image.shape[0]}"
)


# ============================================================
# 4. CONVERT RGB/BGR TO GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 5. HISTOGRAM STRETCHING
# ============================================================

min_pixel = np.min(gray)
max_pixel = np.max(gray)

if max_pixel > min_pixel:

    stretched = (
        (gray.astype(np.float32) - min_pixel)
        * 255.0
        / (max_pixel - min_pixel)
    )

    stretched = np.clip(
        stretched,
        0,
        255
    ).astype(np.uint8)

else:

    stretched = gray.copy()


# ============================================================
# 6. HISTOGRAM EQUALIZATION
# ============================================================

equalized = cv2.equalizeHist(
    gray
)


# ============================================================
# 7. CLAHE
# Contrast Limited Adaptive Histogram Equalization
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(
    gray
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

original_output = (
    OUTPUT_DIR / "original_grayscale.jpg"
)

stretch_output = (
    OUTPUT_DIR / "histogram_stretching.jpg"
)

equalization_output = (
    OUTPUT_DIR / "histogram_equalization.jpg"
)

clahe_output = (
    OUTPUT_DIR / "clahe.jpg"
)


cv2.imwrite(
    str(original_output),
    gray
)

cv2.imwrite(
    str(stretch_output),
    stretched
)

cv2.imwrite(
    str(equalization_output),
    equalized
)

cv2.imwrite(
    str(clahe_output),
    clahe_image
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

cv2.namedWindow(
    "Original Grayscale",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Histogram Stretching",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Histogram Equalization",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "CLAHE",
    cv2.WINDOW_NORMAL
)


cv2.imshow(
    "Original Grayscale",
    gray
)

cv2.imshow(
    "Histogram Stretching",
    stretched
)

cv2.imshow(
    "Histogram Equalization",
    equalized
)

cv2.imshow(
    "CLAHE",
    clahe_image
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print()
print("=" * 60)
print("Histogram Enhancement Completed")
print("=" * 60)

print()
print("Techniques applied:")
print("1. Grayscale conversion")
print("2. Histogram stretching")
print("3. Histogram equalization")
print("4. CLAHE")

print()
print("Output files:")
print(original_output)
print(stretch_output)
print(equalization_output)
print(clahe_output)

print()
print("Press any key in an image window to close all windows.")


# ============================================================
# 11. WAIT AND CLOSE
# ============================================================

cv2.waitKey(0)

cv2.destroyAllWindows()

print()
print("Finished.")