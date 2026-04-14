# 🍎 Advanced AI Project — Hybrid Grading System

> A fruit quality grading system combining **Robust** Software Engineering practices with **Deep Learning** (YOLOv10 + OpenCV XAI).

---

## 📁 Project Structure

```text
advanced-ai-project/
│
├── .github/                       # CI/CD Automation
│   └── workflows/
│       └── python-app.yml         # Auto PEP8 lint & run tests on push
│
├── data/                          # Data (must be in .gitignore)
│   ├── raw/                       # Raw data from Kaggle
│   ├── processed/                 # Annotated data (images + .txt labels)
│   └── dataset.yaml               # Dataset path config for YOLOv10
│
├── models/                        # Model weight storage (.pt files)
│   ├── pretrained/                # Original YOLOv10 pretrained weights
│   └── weights/                   # Training outputs (best.pt, last.pt)
│
├── src/                           # CORE SOURCE CODE (modularized)
│   ├── __init__.py
│   ├── detection/                 # YOLOv10 detection module
│   │   ├── __init__.py
│   │   └── detector.py            # Class to load model and run predictions
│   ├── grading/                   # Hybrid grading logic (OpenCV)
│   │   ├── __init__.py
│   │   └── grader.py              # Color %, size calculation & A/B/C grading
│   ├── explainability/            # XAI module (required for Task 4)
│   │   ├── __init__.py
│   │   └── explainer.py           # Heatmap generation (Grad-CAM/EigenCAM)
│   └── utils/                     # Helper functions (image processing, logs)
│       ├── __init__.py
│       └── image_helpers.py
│
├── notebooks/                     # R&D and Analysis (for demos)
│   ├── 01_data_annotation.ipynb   # Auto-annotation script
│   ├── 02_model_training.ipynb    # Training pipeline & loss/mAP charts
│   └── 03_xai_visualization.ipynb # Model explainability demo
│
├── tests/                         # Quality assurance (robustness)
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_grading.py            # Tests for A/B/C grading logic
│
├── .gitignore                     # Files excluded from Git (data, heavy models)
├── pyproject.toml                 # Code quality tool config (Black, Ruff)
├── requirements.txt               # Required Python packages
└── README.md                      # KEY DOCUMENTATION (roadmap, setup guide)
```

## 🚀 Getting Started

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. GPU Setup (CUDA)

This project requires PyTorch with CUDA support for GPU-accelerated training.
Install the CUDA 12.4 build (**required for RTX 30/40 series**):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** If you have a different CUDA version, check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct install command.

### 3. Code formatting & linting

This project enforces strict PEP8 compliance:

```bash
ruff check src/ tests/
pytest tests/
```

### 4. R&D Workflow

Use the `notebooks/` directory for the research pipeline:

1. Annotate the raw dataset: Run `01_data_annotation.ipynb`
2. Train the YOLOv10 model: Run `02_model_training.ipynb`
3. Inspect with XAI module: Run `03_xai_visualization.ipynb`
