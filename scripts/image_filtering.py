from pathlib import Path
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "Images" / "front" / "IMG_4159.JPG"

OUTPUT_DIR = PROJECT_ROOT / "processed_images" / "image_filtering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")


print("=" * 60)
print("Driver Monitoring System - Image Filtering")
print("=" * 60)

print(f"Input image : {IMAGE_PATH}")
print(f"Output folder: {OUTPUT_DIR}")


# ============================================================
# 1. MEAN / AVERAGE FILTER
# ============================================================

mean_filter = cv2.blur(image, (5, 5))

cv2.imwrite(
    str(OUTPUT_DIR / "01_mean_filter.jpg"),
    mean_filter
)


# ============================================================
# 2. GAUSSIAN FILTER
# ============================================================

gaussian_filter = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_gaussian_filter.jpg"),
    gaussian_filter
)


# ============================================================
# 3. MEDIAN FILTER
# ============================================================

median_filter = cv2.medianBlur(
    image,
    5
)

cv2.imwrite(
    str(OUTPUT_DIR / "03_median_filter.jpg"),
    median_filter
)


# ============================================================
# 4. BILATERAL FILTER
# ============================================================

bilateral_filter = cv2.bilateralFilter(
    image,
    9,
    75,
    75
)

cv2.imwrite(
    str(OUTPUT_DIR / "04_bilateral_filter.jpg"),
    bilateral_filter
)


# ============================================================
# SAVE ORIGINAL
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "00_original.jpg"),
    image
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

cv2.imshow("Original", image)
cv2.imshow("Mean Filter", mean_filter)
cv2.imshow("Gaussian Filter", gaussian_filter)
cv2.imshow("Median Filter", median_filter)
cv2.imshow("Bilateral Filter", bilateral_filter)


print()
print("Filtering completed successfully.")
print()
print("Generated files:")
print("  00_original.jpg")
print("  01_mean_filter.jpg")
print("  02_gaussian_filter.jpg")
print("  03_median_filter.jpg")
print("  04_bilateral_filter.jpg")
print()
print("Press any key in an image window to close all windows.")

cv2.waitKey(0)
cv2.destroyAllWindows()