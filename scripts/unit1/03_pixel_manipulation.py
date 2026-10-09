import cv2
import numpy as np
import matplotlib.pyplot as plt

from scripts.config import UNIT1_RESULTS


def get_4_neighborhood(image, x, y):
    """
    Return the 4-neighborhood pixels:
    Up, Down, Left, Right
    """
    height, width = image.shape[:2]

    neighbors = {}

    if y > 0:
        neighbors["up"] = image[y - 1, x]

    if y < height - 1:
        neighbors["down"] = image[y + 1, x]

    if x > 0:
        neighbors["left"] = image[y, x - 1]

    if x < width - 1:
        neighbors["right"] = image[y, x + 1]

    return neighbors


def get_8_neighborhood(image, x, y):
    """
    Return the 8-neighborhood pixels:
    Up, Down, Left, Right and four diagonal pixels.
    """
    height, width = image.shape[:2]

    neighbors = {}

    positions = {
        "up": (y - 1, x),
        "down": (y + 1, x),
        "left": (y, x - 1),
        "right": (y, x + 1),
        "up_left": (y - 1, x - 1),
        "up_right": (y - 1, x + 1),
        "down_left": (y + 1, x - 1),
        "down_right": (y + 1, x + 1),
    }

    for name, (ny, nx) in positions.items():
        if 0 <= ny < height and 0 <= nx < width:
            neighbors[name] = image[ny, nx]

    return neighbors


def pixel_to_text(pixel):
    """
    Convert a pixel value into readable text.
    """
    if np.isscalar(pixel):
        return str(int(pixel))

    return str(pixel.tolist())


def main():

    print("=" * 75)
    print("UNIT I - PIXEL MANIPULATION & PIXEL RELATIONSHIPS")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. LOAD IMAGE
    # ---------------------------------------------------------

    input_path = UNIT1_RESULTS / "01_acquired_driver_image.jpg"

    if not input_path.exists():

        print()
        print("ERROR: Acquired driver image not found.")
        print(f"Expected file: {input_path}")
        print()
        print("Run Module 01 first.")
        return

    image = cv2.imread(str(input_path))

    if image is None:

        print()
        print("ERROR: Could not load image.")
        return

    height, width, channels = image.shape

    print()
    print("INPUT IMAGE")
    print("-" * 75)
    print(f"File name       : {input_path.name}")
    print(f"Width           : {width} pixels")
    print(f"Height          : {height} pixels")
    print(f"Channels        : {channels}")
    print(f"Data type       : {image.dtype}")

    # ---------------------------------------------------------
    # 2. SELECT CENTER PIXEL
    # ---------------------------------------------------------

    x = width // 2
    y = height // 2

    center_pixel = image[y, x]

    print()
    print("SELECTED PIXEL")
    print("-" * 75)
    print(f"X coordinate    : {x}")
    print(f"Y coordinate    : {y}")
    print(f"Pixel position  : image[{y}, {x}]")
    print(f"BGR value       : {center_pixel}")

    # ---------------------------------------------------------
    # 3. INDIVIDUAL CHANNEL ACCESS
    # ---------------------------------------------------------

    blue_value = int(image[y, x, 0])
    green_value = int(image[y, x, 1])
    red_value = int(image[y, x, 2])

    print()
    print("INDIVIDUAL CHANNEL VALUES")
    print("-" * 75)
    print(f"Blue channel    : {blue_value}")
    print(f"Green channel   : {green_value}")
    print(f"Red channel     : {red_value}")

    # ---------------------------------------------------------
    # 4. MODIFY A SINGLE PIXEL
    # ---------------------------------------------------------

    single_pixel_image = image.copy()

    original_pixel = single_pixel_image[y, x].copy()

    # Change center pixel to red in BGR format
    single_pixel_image[y, x] = [0, 0, 255]

    modified_pixel = single_pixel_image[y, x]

    print()
    print("SINGLE PIXEL MANIPULATION")
    print("-" * 75)
    print(f"Original pixel  : {original_pixel}")
    print(f"Modified pixel  : {modified_pixel}")
    print("Modification    : Center pixel changed to RED")

    single_pixel_output = (
        UNIT1_RESULTS / "03_single_pixel_modified.jpg"
    )

    cv2.imwrite(
        str(single_pixel_output),
        single_pixel_image
    )

    # ---------------------------------------------------------
    # 5. BRIGHTNESS INCREASE
    # ---------------------------------------------------------

    brightness_increase = image.astype(np.int16) + 40

    brightness_increase = np.clip(
        brightness_increase,
        0,
        255
    ).astype(np.uint8)

    brightness_increase_output = (
        UNIT1_RESULTS / "03_brightness_increased.jpg"
    )

    cv2.imwrite(
        str(brightness_increase_output),
        brightness_increase
    )

    # ---------------------------------------------------------
    # 6. BRIGHTNESS DECREASE
    # ---------------------------------------------------------

    brightness_decrease = image.astype(np.int16) - 40

    brightness_decrease = np.clip(
        brightness_decrease,
        0,
        255
    ).astype(np.uint8)

    brightness_decrease_output = (
        UNIT1_RESULTS / "03_brightness_decreased.jpg"
    )

    cv2.imwrite(
        str(brightness_decrease_output),
        brightness_decrease
    )

    # ---------------------------------------------------------
    # 7. BLACK AND WHITE PIXEL MANIPULATION
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    binary = np.where(
        gray >= 128,
        255,
        0
    ).astype(np.uint8)

    binary_output = (
        UNIT1_RESULTS / "03_black_white_pixels.jpg"
    )

    cv2.imwrite(
        str(binary_output),
        binary
    )

    # ---------------------------------------------------------
    # 8. 4-NEIGHBORHOOD
    # ---------------------------------------------------------

    neighborhood_4 = get_4_neighborhood(
        image,
        x,
        y
    )

    print()
    print("4-NEIGHBORHOOD")
    print("-" * 75)

    for direction, pixel in neighborhood_4.items():

        print(
            f"{direction:<12}: {pixel_to_text(pixel)}"
        )

    # ---------------------------------------------------------
    # 9. 8-NEIGHBORHOOD
    # ---------------------------------------------------------

    neighborhood_8 = get_8_neighborhood(
        image,
        x,
        y
    )

    print()
    print("8-NEIGHBORHOOD")
    print("-" * 75)

    for direction, pixel in neighborhood_8.items():

        print(
            f"{direction:<12}: {pixel_to_text(pixel)}"
        )

    # ---------------------------------------------------------
    # 10. LOCAL 3x3 NEIGHBORHOOD
    # ---------------------------------------------------------

    gray_y_start = max(0, y - 1)
    gray_y_end = min(height, y + 2)

    gray_x_start = max(0, x - 1)
    gray_x_end = min(width, x + 2)

    local_3x3 = gray[
        gray_y_start:gray_y_end,
        gray_x_start:gray_x_end
    ]

    print()
    print("LOCAL 3x3 GRAYSCALE NEIGHBORHOOD")
    print("-" * 75)
    print(local_3x3)

    # ---------------------------------------------------------
    # 11. LOCAL STATISTICS
    # ---------------------------------------------------------

    local_mean = float(np.mean(local_3x3))
    local_min = int(np.min(local_3x3))
    local_max = int(np.max(local_3x3))

    print()
    print("LOCAL PIXEL STATISTICS")
    print("-" * 75)
    print(f"Local mean     : {local_mean:.2f}")
    print(f"Local minimum  : {local_min}")
    print(f"Local maximum  : {local_max}")

    # ---------------------------------------------------------
    # 12. SAVE 3x3 NEIGHBORHOOD
    # ---------------------------------------------------------

    neighborhood_output = (
        UNIT1_RESULTS / "03_local_3x3_neighborhood.txt"
    )

    with open(
        neighborhood_output,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("LOCAL 3x3 GRAYSCALE NEIGHBORHOOD\n")
        file.write("=" * 40 + "\n")
        file.write(str(local_3x3))
        file.write("\n\n")

        file.write(f"Center X: {x}\n")
        file.write(f"Center Y: {y}\n")
        file.write(f"Local Mean: {local_mean:.2f}\n")
        file.write(f"Local Minimum: {local_min}\n")
        file.write(f"Local Maximum: {local_max}\n")

    # ---------------------------------------------------------
    # 13. VISUALIZATION
    # ---------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb_modified = cv2.cvtColor(
        single_pixel_image,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(16, 10))

    # Original
    plt.subplot(2, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis("off")

    # Modified pixel
    plt.subplot(2, 3, 2)
    plt.imshow(rgb_modified)
    plt.scatter(
        [x],
        [y],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=2
    )
    plt.title("Modified Center Pixel")
    plt.axis("off")

    # Brightness increased
    plt.subplot(2, 3, 3)
    plt.imshow(
        cv2.cvtColor(
            brightness_increase,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Brightness +40")
    plt.axis("off")

    # Brightness decreased
    plt.subplot(2, 3, 4)
    plt.imshow(
        cv2.cvtColor(
            brightness_decrease,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Brightness -40")
    plt.axis("off")

    # Binary
    plt.subplot(2, 3, 5)
    plt.imshow(binary, cmap="gray")
    plt.title("Black / White Pixels")
    plt.axis("off")

    # 3x3 neighborhood
    plt.subplot(2, 3, 6)

    plt.imshow(
        local_3x3,
        cmap="gray",
        interpolation="nearest"
    )

    plt.title("Local 3x3 Neighborhood")

    for row in range(local_3x3.shape[0]):
        for col in range(local_3x3.shape[1]):

            plt.text(
                col,
                row,
                str(int(local_3x3[row, col])),
                ha="center",
                va="center",
                fontsize=10
            )

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    visualization_output = (
        UNIT1_RESULTS /
        "03_pixel_manipulation_comparison.png"
    )

    plt.savefig(
        str(visualization_output),
        dpi=200,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # 14. OUTPUT SUMMARY
    # ---------------------------------------------------------

    print()
    print("SAVED OUTPUTS")
    print("-" * 75)

    print(single_pixel_output)
    print(brightness_increase_output)
    print(brightness_decrease_output)
    print(binary_output)
    print(neighborhood_output)
    print(visualization_output)

    print()
    print("=" * 75)
    print("PIXEL MANIPULATION COMPLETED")
    print("=" * 75)

    plt.show()


if __name__ == "__main__":
    main()