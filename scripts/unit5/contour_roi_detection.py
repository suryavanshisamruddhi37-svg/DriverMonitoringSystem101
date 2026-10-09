from pathlib import Path
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "Images" / "front" / "IMG_4159.JPG"

OUTPUT_DIR = PROJECT_ROOT / "processed_images" / "contour_roi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(
    str(OUTPUT_DIR / "01_grayscale.jpg"),
    gray
)


# ============================================================
# GAUSSIAN BLUR
# ============================================================

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_blurred.jpg"),
    blurred
)


# ============================================================
# OTSU THRESHOLDING
# ============================================================

_, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(OUTPUT_DIR / "03_otsu.jpg"),
    binary
)


# ============================================================
# MORPHOLOGICAL OPENING
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

cv2.imwrite(
    str(OUTPUT_DIR / "04_opening.jpg"),
    opening
)


# ============================================================
# MORPHOLOGICAL CLOSING
# ============================================================

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imwrite(
    str(OUTPUT_DIR / "05_closing.jpg"),
    closing
)


# ============================================================
# CONTOUR DETECTION
# ============================================================

contours, hierarchy = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ============================================================
# DRAW ALL CONTOURS
# ============================================================

contour_image = image.copy()

cv2.drawContours(
    contour_image,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    str(OUTPUT_DIR / "06_all_contours.jpg"),
    contour_image
)


# ============================================================
# FILTER CONTOURS
# ============================================================

roi_image = image.copy()

candidate_count = 0

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    area = w * h

    # Basic size filtering
    if area < 500:
        continue

    if w < 20 or h < 20:
        continue

    if w > image.shape[1] * 0.8:
        continue

    if h > image.shape[0] * 0.8:
        continue

    candidate_count += 1

    cv2.rectangle(
        roi_image,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    cv2.putText(
        roi_image,
        f"ROI {candidate_count}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )


# ============================================================
# SAVE FINAL ROI IMAGE
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "07_candidate_rois.jpg"),
    roi_image
)


print("=" * 60)
print("UNIT 5: CONTOUR AND ROI DETECTION")
print("=" * 60)
print(f"Input image       : {IMAGE_PATH}")
print(f"Total contours    : {len(contours)}")
print(f"Candidate ROIs    : {candidate_count}")
print(f"Output directory  : {OUTPUT_DIR}")
print("=" * 60)