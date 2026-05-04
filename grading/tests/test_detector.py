import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from grading.src.detection.detector import FruitDetector


@patch('grading.src.detection.detector.YOLO')
def test_detector_initialization(mock_yolo):
    """Test that FruitDetector initializes without crashing and loads the model."""
    detector = FruitDetector("mock_path.pt")
    
    # Ensure YOLO was called with the provided path
    mock_yolo.assert_called_once_with("mock_path.pt")
    assert detector.model_path == "mock_path.pt"

@patch('grading.src.detection.detector.YOLO')
def test_crop_image(mock_yolo):
    """Test the numpy slicing logic for cropping bounding boxes."""
    detector = FruitDetector("mock_path.pt")
    
    # Create a dummy 100x100 RGB image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Bbox: x_min=10, y_min=20, x_max=50, y_max=60
    # Expected crop size: height (60-20=40), width (50-10=40)
    bbox = [10, 20, 50, 60]
    
    cropped = detector.crop_image(dummy_image, bbox)
    
    assert cropped.shape == (40, 40, 3)

@patch('grading.src.detection.detector.YOLO')
def test_crop_image_none_bbox(mock_yolo):
    """Test that crop_image returns the original image if bbox is None."""
    detector = FruitDetector(model_path="dummy.pt")
    detector.model = MagicMock()
    
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert np.array_equal(detector.crop_image(dummy_image, None), dummy_image)
