"""
YOLOv10 detection module.
"""

from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Root của grading/ subfolder (advanced_ai_project/grading/)
BASE_DIR = Path(__file__).resolve().parents[2]
# Default model path: grading/models/weights/fruit_disease_v2/weights/best.pt
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"


class FruitDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize and load the YOLOv10 weights.
        """
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        self.model_path = model_path
        # Load YOLO model
        self.model = YOLO(self.model_path)

    def detect(self, image_path: str):
        """
        Run YOLO model on the input image path.
        Returns:
            bbox: [x_min, y_min, x_max, y_max]
            class_name: The string label of the detected object (e.g., 'Apple', 'Orange')
        Returns None, None if nothing is detected.
        """
        results = self.model.predict(source=image_path, conf=0.25, save=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Pick the first/best box
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            class_id = int(boxes[0].cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            return box, class_name

        return None, None

    def detect_from_array(self, image: np.ndarray):
        """
        Run YOLO model on a numpy array (OpenCV BGR image).

        Args:
            image: OpenCV BGR image as numpy array

        Returns:
            bbox: [x_min, y_min, x_max, y_max]
            class_name: The string label of the detected object
            Returns None, None if nothing is detected.
        """
        results = self.model.predict(source=image, conf=0.25, save=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Pick the first/best box
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            class_id = int(boxes[0].cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            return box, class_name

        return None, None

    def crop_image(self, image: np.ndarray, bbox):
        """
        Crop the image using NumPy slicing based on bounding box coordinates.
        bbox format: [x_min, y_min, x_max, y_max]
        """
        if bbox is None:
            return image

        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
