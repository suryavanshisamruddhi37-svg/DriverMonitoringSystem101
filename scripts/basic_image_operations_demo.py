from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_FOLDER = PROJECT_ROOT / "Images" / "front"

RESIZE_SIZE = 224
ROTATION_ANGLE = 10


# --------------------------------------------------
# FIND FIRST IMAGE
# --------------------------------------------------

image_files = []

for file in IMAGE_FOLDER.iterdir():
    if file.is_file() and file.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        image_files.append(file)

if not image_files:
    print("No image found in Images/front/")
    exit()

image_path = sorted(image_files)[0]

print("Using image:")
print(image_path)


# --------------------------------------------------
# READ IMAGE
# --------------------------------------------------

# Pillow handles the MPO/JPEG-based front images safely.
with Image.open(image_path) as img:
    rgb_image = np.array(img.convert("RGB"))

# Convert RGB to OpenCV BGR
original = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


# --------------------------------------------------
# 1. ORIGINAL RGB
# --------------------------------------------------

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

cv2.imshow("1 - Original RGB", original_rgb)


# --------------------------------------------------
# 2. RESIZING
# --------------------------------------------------

resized = cv2.resize(
    original,
    (RESIZE_SIZE, RESIZE_SIZE),
    interpolation=cv2.INTER_AREA
)

cv2.imshow("2 - Resized 224x224", resized)


# --------------------------------------------------
# 3. CROPPING
# --------------------------------------------------

height, width = original.shape[:2]

crop_size = min(width, height)

x = (width - crop_size) // 2
y = (height - crop_size) // 2

cropped = original[
    y:y + crop_size,
    x:x + crop_size
]

cv2.imshow("3 - Center Cropped", cropped)


# --------------------------------------------------
# 4. ROTATION
# --------------------------------------------------

height, width = resized.shape[:2]

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    ROTATION_ANGLE,
    1.0
)

rotated = cv2.warpAffine(
    resized,
    rotation_matrix,
    (width, height)
)

cv2.imshow("4 - Rotated +10 degrees", rotated)


# --------------------------------------------------
# 5. HORIZONTAL FLIP
# --------------------------------------------------

flipped = cv2.flip(resized, 1)

cv2.imshow("5 - Horizontal Flip", flipped)


# --------------------------------------------------
# 6. RGB → GRAYSCALE
# --------------------------------------------------

grayscale = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2GRAY
)

cv2.imshow("6 - Grayscale", grayscale)


# --------------------------------------------------
# 7. RGB → HSV
# --------------------------------------------------

hsv = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2HSV
)

# Convert HSV to RGB only for normal visual display.
hsv_visual = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2RGB
)

cv2.imshow("7 - HSV", hsv_visual)


# --------------------------------------------------
# 8. HUE CHANNEL
# --------------------------------------------------

hue = hsv[:, :, 0]

cv2.imshow("8 - Hue Channel", hue)


# --------------------------------------------------
# 9. SATURATION CHANNEL
# --------------------------------------------------

saturation = hsv[:, :, 1]

cv2.imshow("9 - Saturation Channel", saturation)


# --------------------------------------------------
# 10. VALUE CHANNEL
# --------------------------------------------------

value = hsv[:, :, 2]

cv2.imshow("10 - Value Channel", value)


# --------------------------------------------------
# WAIT
# --------------------------------------------------

print()
print("Image processing demonstration running.")
print("Press any key in an image window to close all windows.")

cv2.waitKey(0)
cv2.destroyAllWindows()