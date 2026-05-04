import cv2
import numpy as np
import pytest

from grading.src.grading.grader import QualityGrader

@pytest.fixture
def grader():
    """Returns a QualityGrader instance."""
    return QualityGrader()

def test_calculate_size_score(grader):
    """Test the min-max scaling logic for fruit size."""
    # Create a dummy mask with exactly 20,000 pixels (100x200)
    mask = np.ones((100, 200), dtype=np.uint8)
    
    profile = {
        "size_config": {"min_area": 10000, "max_area": 50000}
    }
    
    # Calculation: (20000 - 10000) / (50000 - 10000) * 100 = 25.0
    score = grader.calculate_size_score(mask, profile)
    assert score == 25.0

def test_calculate_size_score_clamping(grader):
    """Test if size score correctly clamps at 0 and 100."""
    mask_small = np.ones((50, 50), dtype=np.uint8) # 2500 pixels
    mask_huge = np.ones((500, 500), dtype=np.uint8) # 250000 pixels
    
    profile = {
        "size_config": {"min_area": 10000, "max_area": 50000}
    }
    
    assert grader.calculate_size_score(mask_small, profile) == 0.0
    assert grader.calculate_size_score(mask_huge, profile) == 100.0

def test_shape_conformity(grader):
    """Test solidity calculation for a perfect rectangle."""
    # Create a blank mask and draw a solid rectangle
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 1, thickness=cv2.FILLED)
    
    # A perfect rectangle has a convex hull equal to its area, so solidity = 100%
    score = grader.calculate_shape_conformity(mask)
    assert score == 100.0

def test_process_fruit_too_small(grader):
    """Test if the system rejects images that are too small."""
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    
    result = grader.process_fruit(tiny_img, "Apple_Healthy")
    
    assert result[0] == "Unanalyzable"
    assert "Too Small" in result[2]
