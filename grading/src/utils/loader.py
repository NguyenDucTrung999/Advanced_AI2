from pathlib import Path

from ultralytics import YOLO

# Root của grading/ subfolder (advanced_ai_project/grading/)
BASE_DIR = Path(__file__).resolve().parents[2]
# Path đến model best.pt
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"

# Cache variable
_detector_model = None


def get_detector():
    global _detector_model

    if _detector_model is None:
        model_path = DEFAULT_MODEL_PATH

        print(f"Loading YOLO model from {model_path}")
        _detector_model = YOLO(str(model_path))

    return _detector_model
