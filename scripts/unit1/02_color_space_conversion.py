import cv2
import numpy as np
import matplotlib.pyplot as plt

from scripts.config import UNIT1_RESULTS


# ============================================================
# UNIT I
# COLOR SPACE CONVERSION
#
# BGR/RGB -> GRAYSCALE
# BGR/RGB -> HSV
# ============================================================


def main():

    print("=" * 70)
    print("UNIT I - COLOR SPACE CONVERSION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Locate acquired image
    # --------------------------------------------------------

    input_path = UNIT1_RESULTS / "01_acquired_driver_image.jpg"

    if not input_path.exists():

        print()
        print("ERROR: Acquired image not found.")
        print(f"Expected: {input_path}")
        print()
        print("Run the image acquisition module first.")
        return

    print()
    print("INPUT IMAGE")
    print("-" * 70)
    print(input_path)

    # --------------------------------------------------------
    # 2. Read image
    # --------------------------------------------------------

    bgr_image = cv2.imread(str(input_path))

    if bgr_image is None:

        print("ERROR: Could not read input image.")
        return

    # --------------------------------------------------------
    # 3. BGR -> RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # 4. RGB -> Grayscale
    # --------------------------------------------------------

    gray_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 5. BGR -> HSV
    # --------------------------------------------------------

    hsv_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # 6. Extract HSV channels
    # --------------------------------------------------------

    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    # --------------------------------------------------------
    # 7. Extract RGB channels
    # --------------------------------------------------------

    red_channel = rgb_image[:, :, 0]
    green_channel = rgb_image[:, :, 1]
    blue_channel = rgb_image[:, :, 2]

    # --------------------------------------------------------
    # 8. Print information
    # --------------------------------------------------------

    print()
    print("COLOR SPACE INFORMATION")
    print("-" * 70)

    print(f"Original BGR shape       : {bgr_image.shape}")
    print(f"RGB shape                : {rgb_image.shape}")
    print(f"Grayscale shape          : {gray_image.shape}")
    print(f"HSV shape                : {hsv_image.shape}")

    print()
    print("CHANNEL INFORMATION")
    print("-" * 70)

    print("RGB channels:")
    print(f"  Red                    : {red_channel.shape}")
    print(f"  Green                  : {green_channel.shape}")
    print(f"  Blue                   : {blue_channel.shape}")

    print()
    print("HSV channels:")
    print(f"  Hue                    : {hue_channel.shape}")
    print(f"  Saturation             : {saturation_channel.shape}")
    print(f"  Value                  : {value_channel.shape}")

    # --------------------------------------------------------
    # 9. Statistics
    # --------------------------------------------------------

    print()
    print("IMAGE STATISTICS")
    print("-" * 70)

    print(
        f"Grayscale minimum       : {np.min(gray_image)}"
    )

    print(
        f"Grayscale maximum       : {np.max(gray_image)}"
    )

    print(
        f"Grayscale mean          : {np.mean(gray_image):.2f}"
    )

    print(
        f"Hue range               : "
        f"{np.min(hue_channel)} - {np.max(hue_channel)}"
    )

    print(
        f"Saturation range        : "
        f"{np.min(saturation_channel)} - "
        f"{np.max(saturation_channel)}"
    )

    print(
        f"Value range             : "
        f"{np.min(value_channel)} - "
        f"{np.max(value_channel)}"
    )

    # --------------------------------------------------------
    # 10. Save images
    # --------------------------------------------------------

    original_output = (
        UNIT1_RESULTS / "02_original_color.jpg"
    )

    grayscale_output = (
        UNIT1_RESULTS / "02_grayscale.jpg"
    )

    hsv_output = (
        UNIT1_RESULTS / "02_hsv.jpg"
    )

    red_output = (
        UNIT1_RESULTS / "02_red_channel.jpg"
    )

    green_output = (
        UNIT1_RESULTS / "02_green_channel.jpg"
    )

    blue_output = (
        UNIT1_RESULTS / "02_blue_channel.jpg"
    )

    hue_output = (
        UNIT1_RESULTS / "02_hue_channel.jpg"
    )

    saturation_output = (
        UNIT1_RESULTS / "02_saturation_channel.jpg"
    )

    value_output = (
        UNIT1_RESULTS / "02_value_channel.jpg"
    )

    # OpenCV saves BGR images
    cv2.imwrite(
        str(original_output),
        bgr_image
    )

    cv2.imwrite(
        str(grayscale_output),
        gray_image
    )

    cv2.imwrite(
        str(hsv_output),
        hsv_image
    )

    cv2.imwrite(
        str(red_output),
        red_channel
    )

    cv2.imwrite(
        str(green_output),
        green_channel
    )

    cv2.imwrite(
        str(blue_output),
        blue_channel
    )

    cv2.imwrite(
        str(hue_output),
        hue_channel
    )

    cv2.imwrite(
        str(saturation_output),
        saturation_channel
    )

    cv2.imwrite(
        str(value_output),
        value_channel
    )

    # --------------------------------------------------------
    # 11. Print saved files
    # --------------------------------------------------------

    print()
    print("SAVED OUTPUTS")
    print("-" * 70)

    print(original_output)
    print(grayscale_output)
    print(hsv_output)
    print(red_output)
    print(green_output)
    print(blue_output)
    print(hue_output)
    print(saturation_output)
    print(value_output)

    # --------------------------------------------------------
    # 12. Create RGB + Grayscale + HSV visualization
    # --------------------------------------------------------

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(hue_channel, cmap="hsv")
    plt.title("Hue")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(saturation_channel, cmap="gray")
    plt.title("Saturation")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(value_channel, cmap="gray")
    plt.title("Value")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(hsv_image)
    plt.title("HSV Representation")
    plt.axis("off")

    plt.tight_layout()

    visualization_path = (
        UNIT1_RESULTS /
        "02_color_space_comparison.png"
    )

    plt.savefig(
        str(visualization_path),
        dpi=200,
        bbox_inches="tight"
    )

    print()
    print("Comparison visualization:")
    print(visualization_path)

    plt.show()

    # --------------------------------------------------------
    # 13. Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLOR SPACE CONVERSION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()