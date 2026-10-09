import cv2
import os


# ============================================================
# DRIVER MONITORING SYSTEM
# UNIT 3 - EDGE DETECTION
# ============================================================

# ------------------------------------------------------------
# INPUT IMAGE
# ------------------------------------------------------------

INPUT_IMAGE = r"Images\side angles\WhatsApp Image 2026-08-17 at 15.40.52 (3).jpeg"


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = r"processed_images\edge_detection"


def main():

    # ========================================================
    # START
    # ========================================================

    print("=" * 70)
    print("        DRIVER MONITORING SYSTEM")
    print("        UNIT 3 - EDGE DETECTION")
    print("=" * 70)

    # ========================================================
    # INPUT IMAGE PATH
    # ========================================================

    absolute_input_path = os.path.abspath(INPUT_IMAGE)

    print("\nINPUT IMAGE")
    print("-" * 70)
    print(absolute_input_path)

    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    if not os.path.exists(absolute_input_path):

        print("\nERROR: Input image not found!")
        print("Check the following path:")
        print(absolute_input_path)

        return

    print("Input file exists: YES")

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(absolute_input_path)

    if image is None:

        print("\nERROR: OpenCV could not read the image.")

        return

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    height, width, channels = image.shape

    print("\nIMAGE INFORMATION")
    print("-" * 70)
    print("Width    :", width)
    print("Height   :", height)
    print("Channels :", channels)
    print("Shape    :", image.shape)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    absolute_output_path = os.path.abspath(
        OUTPUT_DIR
    )

    print("\nOUTPUT DIRECTORY")
    print("-" * 70)
    print(absolute_output_path)

    # ========================================================
    # 1. GRAYSCALE CONVERSION
    # ========================================================

    print("\n[1/5] Converting image to grayscale...")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_path = os.path.join(
        OUTPUT_DIR,
        "00_grayscale.jpg"
    )

    cv2.imwrite(
        gray_path,
        gray
    )

    print("Saved:", gray_path)

    # ========================================================
    # 2. GAUSSIAN BLUR
    # ========================================================

    print("\n[2/5] Applying Gaussian blur...")

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    blurred_path = os.path.join(
        OUTPUT_DIR,
        "01_gaussian_blur.jpg"
    )

    cv2.imwrite(
        blurred_path,
        blurred
    )

    print("Saved:", blurred_path)

    # ========================================================
    # 3. SOBEL EDGE DETECTION
    # ========================================================

    print("\n[3/5] Applying Sobel edge detection...")

    # Horizontal edges
    sobel_x = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Vertical edges
    sobel_y = cv2.Sobel(
        blurred,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Convert to 8-bit
    sobel_x = cv2.convertScaleAbs(
        sobel_x
    )

    sobel_y = cv2.convertScaleAbs(
        sobel_y
    )

    # Combine horizontal and vertical edges
    sobel = cv2.addWeighted(
        sobel_x,
        0.5,
        sobel_y,
        0.5,
        0
    )

    sobel_path = os.path.join(
        OUTPUT_DIR,
        "02_sobel_edges.jpg"
    )

    cv2.imwrite(
        sobel_path,
        sobel
    )

    print("Saved:", sobel_path)

    # ========================================================
    # 4. LAPLACIAN EDGE DETECTION
    # ========================================================

    print("\n[4/5] Applying Laplacian edge detection...")

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    laplacian_path = os.path.join(
        OUTPUT_DIR,
        "03_laplacian_edges.jpg"
    )

    cv2.imwrite(
        laplacian_path,
        laplacian
    )

    print("Saved:", laplacian_path)

    # ========================================================
    # 5. CANNY EDGE DETECTION
    # ========================================================

    print("\n[5/5] Applying Canny edge detection...")

    canny = cv2.Canny(
        blurred,
        80,
        180
    )

    canny_path = os.path.join(
        OUTPUT_DIR,
        "04_canny_edges.jpg"
    )

    cv2.imwrite(
        canny_path,
        canny
    )

    print("Saved:", canny_path)

    # ========================================================
    # OUTPUT VERIFICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    output_files = [
        gray_path,
        blurred_path,
        sobel_path,
        laplacian_path,
        canny_path
    ]

    for file_path in output_files:

        if os.path.exists(file_path):

            file_size = os.path.getsize(
                file_path
            )

            print(
                "[OK]   ",
                os.path.basename(file_path),
                f"({file_size:,} bytes)"
            )

        else:

            print(
                "[ERROR]",
                os.path.basename(file_path)
            )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("DISPLAYING RESULTS")
    print("=" * 70)

    print("\nFive windows will open.")
    print("Press any key while an image window is active to close them.")

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    cv2.imshow(
        "01 - Grayscale",
        gray
    )

    # --------------------------------------------------------
    # Gaussian Blur
    # --------------------------------------------------------

    cv2.imshow(
        "02 - Gaussian Blur",
        blurred
    )

    # --------------------------------------------------------
    # Sobel
    # --------------------------------------------------------

    cv2.imshow(
        "03 - Sobel Edges",
        sobel
    )

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    cv2.imshow(
        "04 - Laplacian Edges",
        laplacian
    )

    # --------------------------------------------------------
    # Canny
    # --------------------------------------------------------

    cv2.imshow(
        "05 - Canny Edges",
        canny
    )

    # ========================================================
    # WAIT
    # ========================================================

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 70)
    print("EDGE DETECTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")
    print(absolute_output_path)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()