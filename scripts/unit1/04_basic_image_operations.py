import cv2
import matplotlib.pyplot as plt

from scripts.config import UNIT1_RESULTS


def main():

    print("=" * 75)
    print("UNIT I - BASIC IMAGE OPERATIONS")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. LOAD IMAGE
    # ---------------------------------------------------------

    input_path = UNIT1_RESULTS / "01_acquired_driver_image.jpg"

    if not input_path.exists():

        print()
        print("ERROR: Acquired image not found.")
        print(f"Expected: {input_path}")
        print()
        print("Run Module 01 first.")
        return

    image = cv2.imread(str(input_path))

    if image is None:

        print()
        print("ERROR: Could not load image.")
        return

    height, width = image.shape[:2]

    print()
    print("INPUT IMAGE")
    print("-" * 75)
    print(f"File name       : {input_path.name}")
    print(f"Original width  : {width}")
    print(f"Original height : {height}")
    print(f"Original shape  : {image.shape}")

    # ---------------------------------------------------------
    # 2. RESIZING
    # ---------------------------------------------------------

    resized_image = cv2.resize(
        image,
        (400, 400),
        interpolation=cv2.INTER_AREA
    )

    resized_output = (
        UNIT1_RESULTS / "04_resized_image.jpg"
    )

    cv2.imwrite(
        str(resized_output),
        resized_image
    )

    print()
    print("RESIZING")
    print("-" * 75)
    print(f"Original size   : {width} x {height}")
    print("New size        : 400 x 400")
    print(f"Saved to        : {resized_output}")

    # ---------------------------------------------------------
    # 3. CROPPING
    # ---------------------------------------------------------

    # Crop central region
    crop_width = int(width * 0.60)
    crop_height = int(height * 0.60)

    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    end_x = start_x + crop_width
    end_y = start_y + crop_height

    cropped_image = image[
        start_y:end_y,
        start_x:end_x
    ]

    cropped_output = (
        UNIT1_RESULTS / "04_cropped_image.jpg"
    )

    cv2.imwrite(
        str(cropped_output),
        cropped_image
    )

    print()
    print("CROPPING")
    print("-" * 75)
    print(f"Crop start      : ({start_x}, {start_y})")
    print(f"Crop end        : ({end_x}, {end_y})")
    print(f"Crop size       : {crop_width} x {crop_height}")
    print(f"Saved to        : {cropped_output}")

    # ---------------------------------------------------------
    # 4. HORIZONTAL FLIP
    # ---------------------------------------------------------

    horizontal_flip = cv2.flip(
        image,
        1
    )

    horizontal_output = (
        UNIT1_RESULTS / "04_horizontal_flip.jpg"
    )

    cv2.imwrite(
        str(horizontal_output),
        horizontal_flip
    )

    print()
    print("HORIZONTAL FLIP")
    print("-" * 75)
    print("Flip code       : 1")
    print("Direction       : Left <-> Right")
    print(f"Saved to        : {horizontal_output}")

    # ---------------------------------------------------------
    # 5. VERTICAL FLIP
    # ---------------------------------------------------------

    vertical_flip = cv2.flip(
        image,
        0
    )

    vertical_output = (
        UNIT1_RESULTS / "04_vertical_flip.jpg"
    )

    cv2.imwrite(
        str(vertical_output),
        vertical_flip
    )

    print()
    print("VERTICAL FLIP")
    print("-" * 75)
    print("Flip code       : 0")
    print("Direction       : Top <-> Bottom")
    print(f"Saved to        : {vertical_output}")

    # ---------------------------------------------------------
    # 6. BOTH AXES FLIP
    # ---------------------------------------------------------

    both_flip = cv2.flip(
        image,
        -1
    )

    both_output = (
        UNIT1_RESULTS / "04_both_axes_flip.jpg"
    )

    cv2.imwrite(
        str(both_output),
        both_flip
    )

    print()
    print("BOTH AXES FLIP")
    print("-" * 75)
    print("Flip code       : -1")
    print("Direction       : Horizontal + Vertical")
    print(f"Saved to        : {both_output}")

    # ---------------------------------------------------------
    # 7. ROTATION - 90 CLOCKWISE
    # ---------------------------------------------------------

    rotate_90_clockwise = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

    rotate_90_clockwise_output = (
        UNIT1_RESULTS / "04_rotate_90_clockwise.jpg"
    )

    cv2.imwrite(
        str(rotate_90_clockwise_output),
        rotate_90_clockwise
    )

    # ---------------------------------------------------------
    # 8. ROTATION - 90 COUNTERCLOCKWISE
    # ---------------------------------------------------------

    rotate_90_counterclockwise = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    rotate_90_counterclockwise_output = (
        UNIT1_RESULTS / "04_rotate_90_counterclockwise.jpg"
    )

    cv2.imwrite(
        str(rotate_90_counterclockwise_output),
        rotate_90_counterclockwise
    )

    # ---------------------------------------------------------
    # 9. ROTATION - 180
    # ---------------------------------------------------------

    rotate_180 = cv2.rotate(
        image,
        cv2.ROTATE_180
    )

    rotate_180_output = (
        UNIT1_RESULTS / "04_rotate_180.jpg"
    )

    cv2.imwrite(
        str(rotate_180_output),
        rotate_180
    )

    print()
    print("ROTATION")
    print("-" * 75)
    print(f"90° clockwise        : {rotate_90_clockwise_output}")
    print(
        f"90° counterclockwise : "
        f"{rotate_90_counterclockwise_output}"
    )
    print(f"180° rotation        : {rotate_180_output}")

    # ---------------------------------------------------------
    # 10. CUSTOM 30 DEGREE ROTATION
    # ---------------------------------------------------------

    center = (
        width // 2,
        height // 2
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        30,
        1.0
    )

    rotated_30 = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    rotated_30_output = (
        UNIT1_RESULTS / "04_rotate_30_degree.jpg"
    )

    cv2.imwrite(
        str(rotated_30_output),
        rotated_30
    )

    print()
    print("CUSTOM ROTATION")
    print("-" * 75)
    print("Angle           : 30 degrees")
    print("Scale           : 1.0")
    print(f"Saved to        : {rotated_30_output}")

    # ---------------------------------------------------------
    # 11. CONVERT TO RGB FOR MATPLOTLIB
    # ---------------------------------------------------------

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized_rgb = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    cropped_rgb = cv2.cvtColor(
        cropped_image,
        cv2.COLOR_BGR2RGB
    )

    horizontal_rgb = cv2.cvtColor(
        horizontal_flip,
        cv2.COLOR_BGR2RGB
    )

    vertical_rgb = cv2.cvtColor(
        vertical_flip,
        cv2.COLOR_BGR2RGB
    )

    rotate_90_rgb = cv2.cvtColor(
        rotate_90_clockwise,
        cv2.COLOR_BGR2RGB
    )

    rotate_180_rgb = cv2.cvtColor(
        rotate_180,
        cv2.COLOR_BGR2RGB
    )

    rotated_30_rgb = cv2.cvtColor(
        rotated_30,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------
    # 12. VISUALIZATION
    # ---------------------------------------------------------

    plt.figure(figsize=(16, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(resized_rgb)
    plt.title("Resized 400 × 400")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(cropped_rgb)
    plt.title("Central Crop")
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.imshow(horizontal_rgb)
    plt.title("Horizontal Flip")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(vertical_rgb)
    plt.title("Vertical Flip")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(rotate_90_rgb)
    plt.title("90° Clockwise")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.imshow(rotate_180_rgb)
    plt.title("180° Rotation")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(rotated_30_rgb)
    plt.title("30° Custom Rotation")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(
        both_flip,
        cv2.COLOR_BGR2RGB
    ))
    plt.title("Both Axes Flip")
    plt.axis("off")

    plt.tight_layout()

    visualization_output = (
        UNIT1_RESULTS /
        "04_basic_image_operations_comparison.png"
    )

    plt.savefig(
        str(visualization_output),
        dpi=200,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # 13. FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("SAVED OUTPUTS")
    print("=" * 75)

    print(resized_output)
    print(cropped_output)
    print(horizontal_output)
    print(vertical_output)
    print(both_output)
    print(rotate_90_clockwise_output)
    print(rotate_90_counterclockwise_output)
    print(rotate_180_output)
    print(rotated_30_output)
    print(visualization_output)

    print()
    print("=" * 75)
    print("BASIC IMAGE OPERATIONS COMPLETED")
    print("=" * 75)

    plt.show()


if __name__ == "__main__":
    main()