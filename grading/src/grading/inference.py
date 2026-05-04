"""
Modern Inference API for Fruit Grading using QualityGrader.
Provides unified interface for YOLO detection + CV grading with XAI reporting.
"""

import cv2
import numpy as np

from grading.src.detection.detector import FruitDetector
from grading.src.explainability.explainer import HeatmapExplainer
from grading.src.grading.grader import QualityGrader
from grading.src.utils.loader import get_detector

# Initialize grader singleton
_grader = None
_explainer = None
_detector = None


def get_grader() -> QualityGrader:
    """Get or create QualityGrader singleton."""
    global _grader
    if _grader is None:
        _grader = QualityGrader()
    return _grader


def get_explainer() -> HeatmapExplainer:
    """Get or create HeatmapExplainer singleton."""
    global _explainer
    if _explainer is None:
        model = get_detector()
        _explainer = HeatmapExplainer(model)
    return _explainer


def predict(image_path: str) -> dict:
    """
    Predict fruit quality grade from image file path.

    Args:
        image_path: Path to the image file

    Returns:
        dict: Grading result with grade, metrics, and XAI report
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not load image from {image_path}"}

    return predict_from_array(image, image_path)


def predict_from_array(image: np.ndarray, source_path: str | None = None) -> dict:
    """
    Predict fruit quality grade from numpy array (OpenCV BGR image).

    Args:
        image: OpenCV BGR image as numpy array
        source_path: Optional source path for reference

    Returns:
        dict: Grading result with grade, metrics, and XAI report
    """
    global _detector

    # Initialize detector if needed
    if _detector is None:
        model = get_detector()
        _detector = FruitDetector()
        _detector.model = model

    grader = get_grader()
    explainer = get_explainer()

    # Step 1: YOLO Detection
    bbox, class_name = _detector.detect_from_array(image)

    if bbox is None:
        return {"success": False, "error": "No fruit detected in image", "grade": "Unknown", "metrics": {}}

    # Step 2: Crop fruit region
    cropped = _detector.crop_image(image, bbox)

    # Step 3: Process through QualityGrader pipeline
    # Normalize class name (replace __ with _ as shown in notebooks)
    safe_class_name = class_name.replace("__", "_")
    xai_report, final_image = grader.process_fruit(cropped, safe_class_name)

    # Step 4: Generate XAI heatmap if defective (only for Rotten fruit)
    # This mirrors the notebook 03 flow: heatmap is generated from the FULL
    # original image (not the cropped region), using EigenCAM on 640x640 input.
    heatmap = None
    yolo_prediction = xai_report["metadata"]["yolo_prediction"]
    if yolo_prediction != "Healthy":
        try:
            # Pass the original BGR image directly — explainer handles BGR->RGB
            heatmap = explainer.generate_heatmap(image)
        except Exception as e:
            print(f"[XAI] Heatmap generation skipped: {e}")

    # Build response
    return {
        "success": True,
        "grade": xai_report["decision"]["grade"],
        "class_name": safe_class_name,
        "metrics": xai_report["metrics"],
        "xai": {
            "reasons": xai_report["decision"]["reasons"],
            "metadata": xai_report["metadata"],
            "detection": {"bbox": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox), "class": safe_class_name},
        },
        "heatmap": heatmap,
        "source": source_path,
    }


def predict_batch(image_paths: list) -> list:
    """
    Batch prediction for multiple images.

    Args:
        image_paths: List of image file paths

    Returns:
        list: List of prediction results
    """
    results = []
    for path in image_paths:
        result = predict(path)
        results.append(result)
    return results
