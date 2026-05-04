"""
YOLOv10 detection module.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Root của grading/ subfolder (advanced_ai_project/grading/)
BASE_DIR = Path(__file__).resolve().parents[2]
# Default model path: grading/models/weights/fruit_disease_v2/weights/best.pt
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"


class FruitDetector:
    """YOLOv10-based detector for identifying fruit and their conditions.

    Attributes:
        model_path (str): Path to the loaded YOLO model weights.
        model (YOLO): The loaded YOLOv10 model instance.
    """

    def __init__(self, model_path: str = None):
        """Initializes the FruitDetector and loads the YOLOv10 weights.

        Args:
            model_path (str, optional): Path to the YOLOv10 weights file.
                Defaults to the DEFAULT_MODEL_PATH if None.
        """
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        self.model_path = model_path
        # Load YOLO model
        self.model = YOLO(self.model_path)

    def detect(self, image_path: str):
        """Runs YOLO model prediction on an image file.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            tuple: A tuple containing:
                - box (numpy.ndarray or None): [x_min, y_min, x_max, y_max] coordinates.
                - class_name (str or None): The label of the detected object.
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
        """Runs YOLO model prediction on an image array.

        Args:
            image (numpy.ndarray): OpenCV BGR image as a numpy array.

        Returns:
            tuple: A tuple containing:
                - box (numpy.ndarray or None): [x_min, y_min, x_max, y_max] coordinates.
                - class_name (str or None): The label of the detected object.
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
        """Crops an image based on the provided bounding box.

        Args:
            image (numpy.ndarray): Original image array.
            bbox (list or numpy.ndarray): [x_min, y_min, x_max, y_max] coordinates.

        Returns:
            numpy.ndarray: The cropped image region.
        """
        if bbox is None:
            return image

        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
