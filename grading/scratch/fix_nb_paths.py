"""
Sửa đường dẫn trong các notebook sau khi toàn bộ grading code được
chuyển vào subfolder grading/ bên trong advanced_ai_project/.

Cấu trúc mới:
  advanced_ai_project/
    grading/
      src/grading/grader.py
      models/weights/fruit_disease_v2/weights/best.pt
      data/raw/Fruit And Vegetable Diseases Dataset/
      data/processed/
      data/test_images/
    notebooks/   ← vẫn ở đây, PROJECT_ROOT = Path("..") = advanced_ai_project/
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[2] / "notebooks"


def load_nb(name):
    with open(NB_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb, name):
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def fix_cell_source(source_list: list, replacements: list) -> tuple[list, int]:
    """Apply string replacements to a cell's source list. Returns (new_source, count_changed)."""
    src = "".join(source_list)
    changed = 0
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            changed += 1
    if changed:
        lines = src.split("\n")
        new_source = [l + "\n" for l in lines]
        new_source[-1] = new_source[-1].rstrip("\n")
        return new_source, changed
    return source_list, 0


# ─── Replacements chung cho notebooks 01, 02, 03, 04 ────────────────────────
# Tất cả path dưới đây trước đây trỏ thẳng vào advanced_ai_project/xxx
# Giờ phải qua grading/ trước: advanced_ai_project/grading/xxx

COMMON_REPLACEMENTS = [
    # src imports
    ('from src.grading.grader import QualityGrader',
     'from grading.src.grading.grader import QualityGrader'),

    ('from src.utils.download_dataset import download_dataset',
     'from grading.src.utils.download_dataset import download_dataset'),

    # model path
    ('MODEL_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"',
     'MODEL_WEIGHTS = PROJECT_ROOT / "grading" / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"'),

    ('WEIGHTS_PATH = PROJECT_ROOT / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"',
     'WEIGHTS_PATH = PROJECT_ROOT / "grading" / "models" / "weights" / "fruit_disease_v2" / "weights" / "best.pt"'),

    # data paths
    ('TEST_IMG_DIR = PROJECT_ROOT / "data" / "test_images"',
     'TEST_IMG_DIR = PROJECT_ROOT / "grading" / "data" / "test_images"'),

    ('TEST_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"',
     'TEST_IMG_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"'),

    ('RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"',
     'RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"'),

    ('RAW_DIR = PROJECT_ROOT / "data" / "raw"',
     'RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw"'),

    ('PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"',
     'PROCESSED_DIR = PROJECT_ROOT / "grading" / "data" / "processed"'),

    # notebook 04 - RAW_DIR reference
    ('RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"',
     'RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"'),  # no-op guard

    # notebook 02 - dataset yaml and dirs
    ('DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"',
     'DATASET_YAML = PROJECT_ROOT / "grading" / "data" / "processed" / "dataset.yaml"'),

    ('PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"',
     'PRETRAINED_DIR = PROJECT_ROOT / "grading" / "models" / "pretrained"'),

    ('WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"',
     'WEIGHTS_DIR = PROJECT_ROOT / "grading" / "models" / "weights"'),
]

# notebook 04 specific: RAW_DIR variable used in auto loop
NB04_EXTRA = [
    ('RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"\n'
     '\nprint',
     'RAW_DIR = PROJECT_ROOT / "grading" / "data" / "raw" / "Fruit And Vegetable Diseases Dataset"\n'
     '\nprint'),  # no-op, already correct after COMMON

    # Fix assert line in nb04
    ('assert RAW_DIR.exists(), f"[ERROR] Dataset not found: {RAW_DIR}"',
     'assert RAW_DIR.exists(), f"[ERROR] Dataset not found: {RAW_DIR}"'),  # no-op
]


def fix_notebook(name, extra_replacements=None):
    nb = load_nb(name)
    total_changed = 0
    reps = COMMON_REPLACEMENTS + (extra_replacements or [])

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_src, n = fix_cell_source(cell["source"], reps)
        if n:
            cell["source"] = new_src
            total_changed += n

    save_nb(nb, name)
    print(f"  {name}: {total_changed} replacement(s) applied")
    return total_changed


print("=" * 60)
print("Fixing notebook paths for grading/ subfolder structure")
print("=" * 60)

fix_notebook("01_data_annotation.ipynb")
fix_notebook("02_yolov10_model_training.ipynb")
fix_notebook("03_yolov10_xai_visualization.ipynb")
fix_notebook("04_healthy_fruit_grading.ipynb")

print()
print("Verifying — checking for leftover wrong paths...")
bad_patterns = [
    'PROJECT_ROOT / "src"',
    'PROJECT_ROOT / "models"',
    'PROJECT_ROOT / "data"',
    'from src.',
    'sys.path.append(str(PROJECT_ROOT / "apps"',
    '"ai_models"',
]

all_ok = True
for nb_name in ["01_data_annotation.ipynb", "02_yolov10_model_training.ipynb",
                "03_yolov10_xai_visualization.ipynb", "04_healthy_fruit_grading.ipynb"]:
    nb = load_nb(nb_name)
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    found = [p for p in bad_patterns if p in all_src]
    if found:
        print(f"  [!!] {nb_name} still has: {found}")
        all_ok = False
    else:
        print(f"  [OK] {nb_name}")

print()
print("Done!" if all_ok else "Some issues remain — check above.")
