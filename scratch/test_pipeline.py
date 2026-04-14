import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"d:\in-class\Advance_AI2\advanced_ai_project").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.detection.detector import FruitDetector
from src.grading.grader import QualityGrader
from src.utils.color_analyzer import auto_find_rot_hsv

model_path = PROJECT_ROOT / "models" / "weights" / "fruit_disease_v1" / "weights" / "best.pt"
detector = FruitDetector(str(model_path))
grader = QualityGrader()

img_path = PROJECT_ROOT / "data" / "test_images" / "tao_hong_1.jpg"
print(f"File exists: {img_path.exists()}")

image = cv2.imread(str(img_path))
bbox, class_name = detector.detect(str(img_path))
print(f"bbox: {bbox}, class: {class_name}")

cropped_fruit = detector.crop_image(image, bbox)
print(f"cropped shape: {cropped_fruit.shape}")

img_no_bg, fruit_mask = grader.remove_background(cropped_fruit)
print("grabcut done")

lower_hsv, upper_hsv = auto_find_rot_hsv(img_no_bg, k=3)
print(f"hsv: {lower_hsv}, {upper_hsv}")

fruit_area, rot_area, highlighted_img, mask_raw, mask_clean, extracted_rot = grader.find_defects(img_no_bg, fruit_mask, lower_hsv, upper_hsv)
print(f"areas: {fruit_area}, {rot_area}")
