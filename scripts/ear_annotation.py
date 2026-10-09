

from pathlib import Path
import cv2


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = PROJECT_ROOT / "dataset"

# We will initially annotate side-profile and diagonal images.
SOURCE_FOLDERS = [
    "left_side",
    "right_side",
    "left_diagonal",
    "right_diagonal",
]

ANNOTATION_ROOT = PROJECT_ROOT / "ear_dataset"

IMAGE_OUTPUT = ANNOTATION_ROOT / "images"
LABEL_OUTPUT = ANNOTATION_ROOT / "labels"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

WINDOW_NAME = "Ear Annotation Tool"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ---------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------

drawing = False
start_point = None
current_point = None
image = None
display_image = None


# ---------------------------------------------------------
# MOUSE CALLBACK
# ---------------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_point
    global current_point
    global display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            current_point = (x, y)

            display_image = image.copy()

            cv2.rectangle(
                display_image,
                start_point,
                current_point,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                display_image
            )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        current_point = (x, y)

        display_image = image.copy()

        cv2.rectangle(
            display_image,
            start_point,
            current_point,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            display_image
        )


# ---------------------------------------------------------
# GET IMAGES
# ---------------------------------------------------------

def collect_images():

    images = []

    for folder_name in SOURCE_FOLDERS:

        folder = SOURCE_ROOT / folder_name

        if not folder.exists():
            continue

        for file in folder.iterdir():

            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                images.append(file)

    return sorted(images)


# ---------------------------------------------------------
# YOLO CONVERSION
# ---------------------------------------------------------

def convert_to_yolo(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
):

    # Make coordinates ordered.
    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    # Bounding-box center.
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    width = right - left
    height = bottom - top

    # Normalize to 0-1.
    center_x /= image_width
    center_y /= image_height

    width /= image_width
    height /= image_height

    return center_x, center_y, width, height


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    global image
    global display_image

    ANNOTATION_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    LABEL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    images = collect_images()

    print("=" * 60)
    print("Driver Monitoring System - Ear Annotation")
    print("=" * 60)

    print(f"Images to annotate: {len(images)}")
    print()

    print("Instructions:")
    print("  LEFT MOUSE DRAG = draw bounding box around visible ear")
    print("  S = save annotation")
    print("  N = skip image")
    print("  Q = quit")
    print()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    for index, image_path in enumerate(images, start=1):

        print()
        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print("Could not read image.")
            continue

        display_image = image.copy()

        cv2.imshow(
            WINDOW_NAME,
            display_image
        )

        while True:

            key = cv2.waitKey(0) & 0xFF

            # -----------------------------------------
            # SAVE
            # -----------------------------------------

            if key == ord("s"):

                if (
                    start_point is None
                    or current_point is None
                ):

                    print(
                        "Draw a bounding box first."
                    )

                    continue

                height, width = image.shape[:2]

                cx, cy, bw, bh = convert_to_yolo(
                    start_point[0],
                    start_point[1],
                    current_point[0],
                    current_point[1],
                    width,
                    height
                )

                # YOLO class 0 = ear
                annotation = (
                    f"0 {cx:.6f} {cy:.6f} "
                    f"{bw:.6f} {bh:.6f}\n"
                )

                # Copy image.
                output_image = (
                    IMAGE_OUTPUT
                    / image_path.name
                )

                cv2.imwrite(
                    str(output_image),
                    image
                )

                # Save annotation.
                output_label = (
                    LABEL_OUTPUT
                    / f"{image_path.stem}.txt"
                )

                output_label.write_text(
                    annotation,
                    encoding="utf-8"
                )

                print(
                    f"Saved ear annotation: "
                    f"{output_label.name}"
                )

                break

            # -----------------------------------------
            # SKIP
            # -----------------------------------------

            elif key == ord("n"):

                print("Skipped.")

                break

            # -----------------------------------------
            # QUIT
            # -----------------------------------------

            elif key == ord("q"):

                print(
                    "Annotation stopped."
                )

                cv2.destroyAllWindows()

                return

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("Ear annotation completed.")
    print("=" * 60)

    print(
        f"Images: {IMAGE_OUTPUT}"
    )

    print(
        f"Labels: {LABEL_OUTPUT}"
    )


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()