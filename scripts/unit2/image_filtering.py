import cv2
import os


# ============================================================
# DRIVER MONITORING SYSTEM
# UNIT 2 - IMAGE FILTERING AND NOISE REMOVAL
# ============================================================

# Input image
INPUT_IMAGE = r"Images\side angles\WhatsApp Image 2026-08-17 at 15.40.52 (3).jpeg"

# Output directory
OUTPUT_DIR = r"processed_images\image_filtering"


def main():

    print("=" * 60)
    print("Driver Monitoring System - Image Filtering")
    print("=" * 60)

    # --------------------------------------------------------
    # Get absolute input path
    # --------------------------------------------------------
    absolute_input_path = os.path.abspath(INPUT_IMAGE)

    print("\nInput image:")
    print(absolute_input_path)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------
    if not os.path.exists(absolute_input_path):
        print("\nERROR: Input image not found!")
        print("Check this path:")
        print(absolute_input_path)
        return

    print("Input file exists: YES")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------
    image = cv2.imread(absolute_input_path)

    if image is None:
        print("\nERROR: OpenCV could not read the image.")
        return

    height, width, channels = image.shape

    print("\nImage information:")
    print("Width    :", width)
    print("Height   :", height)
    print("Channels :", channels)
    print("Shape    :", image.shape)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    absolute_output_path = os.path.abspath(OUTPUT_DIR)

    print("\nOutput folder:")
    print(absolute_output_path)

    # ========================================================
    # ORIGINAL
    # ========================================================

    original_path = os.path.join(
        OUTPUT_DIR,
        "00_original.jpg"
    )

    cv2.imwrite(original_path, image)

    # ========================================================
    # MEAN FILTER
    # ========================================================

    mean_filter = cv2.blur(
        image,
        (5, 5)
    )

    mean_path = os.path.join(
        OUTPUT_DIR,
        "01_mean_filter.jpg"
    )

    cv2.imwrite(mean_path, mean_filter)

    # ========================================================
    # GAUSSIAN FILTER
    # ========================================================

    gaussian_filter = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    gaussian_path = os.path.join(
        OUTPUT_DIR,
        "02_gaussian_filter.jpg"
    )

    cv2.imwrite(gaussian_path, gaussian_filter)

    # ========================================================
    # MEDIAN FILTER
    # ========================================================

    median_filter = cv2.medianBlur(
        image,
        5
    )

    median_path = os.path.join(
        OUTPUT_DIR,
        "03_median_filter.jpg"
    )

    cv2.imwrite(median_path, median_filter)

    # ========================================================
    # BILATERAL FILTER
    # ========================================================

    bilateral_filter = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    bilateral_path = os.path.join(
        OUTPUT_DIR,
        "04_bilateral_filter.jpg"
    )

    cv2.imwrite(bilateral_path, bilateral_filter)

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    print("\nFiltering completed successfully.")

    print("\nGenerated files:")

    output_files = [
        original_path,
        mean_path,
        gaussian_path,
        median_path,
        bilateral_path
    ]

    for file_path in output_files:

        if os.path.exists(file_path):

            print(
                "  [OK]",
                os.path.basename(file_path)
            )

        else:

            print(
                "  [ERROR]",
                os.path.basename(file_path)
            )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\nOpening result windows...")
    print("Press any key inside an image window to close them.")

    cv2.imshow(
        "01 - ORIGINAL",
        image
    )

    cv2.imshow(
        "02 - MEAN FILTER",
        mean_filter
    )

    cv2.imshow(
        "03 - GAUSSIAN FILTER",
        gaussian_filter
    )

    cv2.imshow(
        "04 - MEDIAN FILTER",
        median_filter
    )

    cv2.imshow(
        "05 - BILATERAL FILTER",
        bilateral_filter
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("IMAGE FILTERING COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()