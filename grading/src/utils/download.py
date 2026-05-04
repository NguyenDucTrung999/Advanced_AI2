import shutil
from pathlib import Path

import kagglehub

# Root của grading/ subfolder (advanced_ai_project/grading/)
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "cazofi/fruit-disease-v2/pyTorch/default"
# Path đến thư mục chứa best.pt
MODEL_DIR = BASE_DIR / "models" / "weights" / "fruit_disease_v2" / "weights"


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading model from Kaggle...")

    try:
        path = Path(kagglehub.model_download(MODEL_NAME))
        print(f"Kaggle path: {path}")

        # Check if the cache directory is empty (common if 'move' was used previously)
        files = list(path.glob("*"))
        if not files:
            print("Cache is empty! Clearing and forcing re-download...")
            shutil.rmtree(path)
            path = Path(kagglehub.model_download(MODEL_NAME))
            files = list(path.glob("*"))

        if not files:
            print("Warning: No files found in the model download.")
            return

        # Copy files to your model directory
        for file in files:
            print(f"Copying {file.name} to project...")
            shutil.copy(str(file), MODEL_DIR / file.name)

        print("Model ready at:", MODEL_DIR)

    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


if __name__ == "__main__":
    download_model()
