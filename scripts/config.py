from pathlib import Path


# ============================================================
# DRIVER MONITORING SYSTEM
# GLOBAL PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# INPUT DATA
# ------------------------------------------------------------

IMAGE_ROOT = PROJECT_ROOT / "Images"
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_SPLIT_ROOT = PROJECT_ROOT / "dataset_split"

EAR_DATASET_ROOT = PROJECT_ROOT / "ear_dataset"
EAR_YOLO_DATASET_ROOT = PROJECT_ROOT / "ear_yolo_dataset"


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "results"

UNIT1_RESULTS = RESULTS_ROOT / "unit1"
UNIT2_RESULTS = RESULTS_ROOT / "unit2"
UNIT3_RESULTS = RESULTS_ROOT / "unit3"
UNIT4_RESULTS = RESULTS_ROOT / "unit4"
DMS_RESULTS = RESULTS_ROOT / "driver_monitoring"


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

MODELS_ROOT = PROJECT_ROOT / "models"


# ------------------------------------------------------------
# IMAGE SETTINGS
# ------------------------------------------------------------

TARGET_SIZE = (224, 224)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ------------------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ------------------------------------------------------------

for directory in [
    UNIT1_RESULTS,
    UNIT2_RESULTS,
    UNIT3_RESULTS,
    UNIT4_RESULTS,
    DMS_RESULTS,
    MODELS_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# TEST CONFIGURATION
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DRIVER MONITORING SYSTEM CONFIGURATION")
    print("=" * 60)

    print("Project Root :", PROJECT_ROOT)
    print("Images       :", IMAGE_ROOT)
    print("Dataset      :", DATASET_ROOT)
    print("Ear Dataset  :", EAR_DATASET_ROOT)
    print("Results      :", RESULTS_ROOT)
    print("Models       :", MODELS_ROOT)

    print("=" * 60)