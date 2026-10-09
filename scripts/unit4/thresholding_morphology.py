import cv2
import os


# ============================================================
# DRIVER MONITORING SYSTEM
# UNIT 4 - THRESHOLDING AND MORPHOLOGICAL PROCESSING
# ============================================================

# ------------------------------------------------------------
# INPUT IMAGE
# ------------------------------------------------------------

INPUT_IMAGE = r"Images\side angles\WhatsApp Image 2026-08-17 at 15.40.52 (3).jpeg"


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = r"processed_images\thresholding_morphology"


def main():

    # ========================================================
    # START
    # ========================================================

    print("=" * 70)
    print("        DRIVER MONITORING SYSTEM")
    print("        UNIT 4 - THRESHOLDING AND MORPHOLOGY")
    print("=" * 70)

    # ========================================================
    # INPUT PATH
    # ========================================================

    absolute_input_path = os.path.abspath(INPUT_IMAGE)

    print("\nINPUT IMAGE")
    print("-" * 70)
    print(absolute_input_path)

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not os.path.exists(absolute_input_path):

        print("\nERROR: Input image not found!")
        print(absolute_input_path)

        return

    print("Input file exists: YES")

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        absolute_input_path
    )

    if image is None:

        print("\nERROR: Could not read image.")

        return

    height, width, channels = image.shape

    print("\nIMAGE INFORMATION")
    print("-" * 70)
    print("Width    :", width)
    print("Height   :", height)
    print("Channels :", channels)

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
    # 1. GRAYSCALE
    # ========================================================

    print("\n[1/8] Converting to grayscale...")

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

    # ========================================================
    # 2. GLOBAL THRESHOLD
    # ========================================================

    print("[2/8] Applying global threshold...")

    _, global_threshold = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    global_path = os.path.join(
        OUTPUT_DIR,
        "01_global_threshold.jpg"
    )

    cv2.imwrite(
        global_path,
        global_threshold
    )

    # ========================================================
    # 3. OTSU THRESHOLD
    # ========================================================

    print("[3/8] Applying Otsu threshold...")

    otsu_value, otsu_threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_path = os.path.join(
        OUTPUT_DIR,
        "02_otsu_threshold.jpg"
    )

    cv2.imwrite(
        otsu_path,
        otsu_threshold
    )

    print("Otsu threshold value:", otsu_value)

    # ========================================================
    # 4. ADAPTIVE THRESHOLD
    # ========================================================

    print("[4/8] Applying adaptive threshold...")

    adaptive_threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    adaptive_path = os.path.join(
        OUTPUT_DIR,
        "03_adaptive_threshold.jpg"
    )

    cv2.imwrite(
        adaptive_path,
        adaptive_threshold
    )

    # ========================================================
    # MORPHOLOGICAL KERNEL
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # ========================================================
    # 5. EROSION
    # ========================================================

    print("[5/8] Applying erosion...")

    erosion = cv2.erode(
        otsu_threshold,
        kernel,
        iterations=1
    )

    erosion_path = os.path.join(
        OUTPUT_DIR,
        "04_erosion.jpg"
    )

    cv2.imwrite(
        erosion_path,
        erosion
    )

    # ========================================================
    # 6. DILATION
    # ========================================================

    print("[6/8] Applying dilation...")

    dilation = cv2.dilate(
        otsu_threshold,
        kernel,
        iterations=1
    )

    dilation_path = os.path.join(
        OUTPUT_DIR,
        "05_dilation.jpg"
    )

    cv2.imwrite(
        dilation_path,
        dilation
    )

    # ========================================================
    # 7. OPENING
    # ========================================================

    print("[7/8] Applying morphological opening...")

    opening = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    opening_path = os.path.join(
        OUTPUT_DIR,
        "06_opening.jpg"
    )

    cv2.imwrite(
        opening_path,
        opening
    )

    # ========================================================
    # 8. CLOSING
    # ========================================================

    print("[8/8] Applying morphological closing...")

    closing = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    closing_path = os.path.join(
        OUTPUT_DIR,
        "07_closing.jpg"
    )

    cv2.imwrite(
        closing_path,
        closing
    )

    # ========================================================
    # VERIFY OUTPUTS
    # ========================================================

    print("\n" + "=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    output_files = [
        gray_path,
        global_path,
        otsu_path,
        adaptive_path,
        erosion_path,
        dilation_path,
        opening_path,
        closing_path
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

    cv2.imshow(
        "01 - Grayscale",
        gray
    )

    cv2.imshow(
        "02 - Global Threshold",
        global_threshold
    )

    cv2.imshow(
        "03 - Otsu Threshold",
        otsu_threshold
    )

    cv2.imshow(
        "04 - Adaptive Threshold",
        adaptive_threshold
    )

    cv2.imshow(
        "05 - Erosion",
        erosion
    )

    cv2.imshow(
        "06 - Dilation",
        dilation
    )

    cv2.imshow(
        "07 - Opening",
        opening
    )

    cv2.imshow(
        "08 - Closing",
        closing
    )

    print("\nEight result windows opened.")
    print("Press any key inside an image window to close them.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 70)
    print("UNIT 4 - THRESHOLDING AND MORPHOLOGY COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print(absolute_output_path)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()