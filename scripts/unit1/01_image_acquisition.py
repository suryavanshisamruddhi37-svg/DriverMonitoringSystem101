import cv2
import numpy as np

from scripts.config import IMAGE_ROOT, UNIT1_RESULTS




VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_front_images():
    """
    Find all valid images inside Images/front.
    """

    front_folder = IMAGE_ROOT / "front"

    if not front_folder.exists():
        print("ERROR: Images/front folder does not exist.")
        return []

    images = sorted(
        [
            path
            for path in front_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ]
    )

    return images


def select_image(images):
    """
    Ask the user to select an image manually.
    """

    print()
    print("=" * 70)
    print("AVAILABLE DRIVER IMAGES")
    print("=" * 70)

    for index, image_path in enumerate(images, start=1):
        print(f"{index:3d}. {image_path.name}")

    print("=" * 70)

    while True:

        try:
            choice = int(
                input(
                    f"\nSelect image number (1-{len(images)}): "
                )
            )

            if 1 <= choice <= len(images):
                return images[choice - 1]

            print(
                f"Enter a number between 1 and {len(images)}."
            )

        except ValueError:
            print("Please enter a valid number.")


def main():

    print("=" * 70)
    print("UNIT I - IMAGE ACQUISITION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Find images
    # --------------------------------------------------------

    images = get_front_images()

    if not images:
        print("No images found.")
        return

    print()
    print(f"Found {len(images)} images in:")
    print(IMAGE_ROOT / "front")

    # --------------------------------------------------------
    # 2. Select image
    # --------------------------------------------------------

    image_path = select_image(images)

    print()
    print("SELECTED IMAGE")
    print("-" * 70)
    print("File name :", image_path.name)
    print("Location  :", image_path)

    # --------------------------------------------------------
    # 3. Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not load selected image.")
        return

    # --------------------------------------------------------
    # 4. Image information
    # --------------------------------------------------------

    height, width, channels = image.shape

    total_pixels = width * height
    total_values = image.size

    minimum_value = int(np.min(image))
    maximum_value = int(np.max(image))
    mean_intensity = float(np.mean(image))

    # --------------------------------------------------------
    # 5. Print image representation
    # --------------------------------------------------------

    print()
    print("IMAGE REPRESENTATION")
    print("-" * 70)

    print(f"Width              : {width} pixels")
    print(f"Height             : {height} pixels")
    print(f"Channels           : {channels}")
    print(f"Total pixels       : {total_pixels}")
    print(f"Total values       : {total_values}")
    print(f"Data type          : {image.dtype}")
    print(f"Minimum pixel      : {minimum_value}")
    print(f"Maximum pixel      : {maximum_value}")
    print(f"Mean intensity     : {mean_intensity:.2f}")

    # --------------------------------------------------------
    # 6. NumPy representation
    # --------------------------------------------------------

    print()
    print("NUMPY MATRIX")
    print("-" * 70)

    print(f"Matrix shape       : {image.shape}")
    print(f"Matrix dimensions  : {image.ndim}")
    print(f"Matrix data type   : {image.dtype}")

    # --------------------------------------------------------
    # 7. Display sample pixel
    # --------------------------------------------------------

    center_x = width // 2
    center_y = height // 2

    center_pixel = image[center_y, center_x]

    print()
    print("SAMPLE PIXEL")
    print("-" * 70)

    print(f"Center X           : {center_x}")
    print(f"Center Y           : {center_y}")
    print(f"BGR pixel value    : {center_pixel}")

    # --------------------------------------------------------
    # 8. Save image
    # --------------------------------------------------------

    output_path = UNIT1_RESULTS / "01_acquired_driver_image.jpg"

    saved = cv2.imwrite(
        str(output_path),
        image
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    if saved:
        print("Saved to:")
        print(output_path)
    else:
        print("ERROR: Could not save image.")

    # --------------------------------------------------------
    # 9. Display image
    # --------------------------------------------------------

    cv2.imshow(
        "UNIT I - Acquired Driver Image",
        image
    )

    print()
    print("Press any key inside the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # 10. Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE ACQUISITION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()