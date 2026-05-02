import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Root của grading/ subfolder (advanced_ai_project/grading/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Đường dẫn dataset
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = BASE_DIR / "data" / "raw"
# Tên dataset trên Kaggle (format: username/dataset-slug)
DATASET = "marquis03/fruit-and-vegetable-disease"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Kaggle credentials from .env via settings ──
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise Exception("Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    print(f"Using Kaggle account: {kaggle_username}")

    # ── Authenticate ──
    api = KaggleApi()
    api.authenticate()

    # ── Skip if already downloaded ──
    if any(RAW_DIR.iterdir()):
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading dataset...")
    api.dataset_download_files(DATASET, path=DATA_DIR, unzip=False)

    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        raise Exception("No dataset zip found after download")

    zip_path = zip_files[0]

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    # Clean up zip file
    zip_path.unlink(missing_ok=True)

    print(f"Dataset ready at: {RAW_DIR}")


if __name__ == "__main__":
    download_dataset()
