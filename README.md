# PCOS Detection using Machine Learning

This repository contains an educational, research-oriented PCOS screening
project. It includes a Streamlit interface, a reproducible ultrasound image
training pipeline, and a small tabular baseline. It is **not a medical
diagnostic tool** and does not replace professional clinical advice.

## What is in the repository?

- `train.py` trains an EfficientNetB0 image classifier from a class-folder dataset.
- `evaluate.py` evaluates a saved Keras classifier on a class-folder dataset.
- `predict.py` runs inference for one image.
- `tabular.py` trains a binary Random Forest baseline from a CSV file.
- `app.py` provides the existing Streamlit user interface when compatible model
  artifacts are supplied under `models/`.
- `data_utils.py` contains deterministic, non-destructive dataset utilities.

No dataset, trained model, or measured performance result is committed here.

## Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/pallavi12-code/PCOS-Detection-using-Machine-Learning.git
cd PCOS-Detection-using-Machine-Learning
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset structure

The image pipeline expects one directory per class. Class names become the
saved label mapping and are sorted alphabetically.

```text
data/ultrasound/
├── normal/
│   ├── image-001.jpg
│   └── image-002.jpg
└── pcos/
    ├── image-101.jpg
    └── image-102.jpg
```

Supported image formats are JPEG, PNG, BMP, TIFF, and WebP. The source dataset
is read only; training never deletes, moves, or rewrites source files.

## Train the image model

```bash
python train.py \
  --data-dir data/ultrasound \
  --output-dir artifacts/image \
  --epochs 10 \
  --batch-size 16 \
  --validation-fraction 0.2 \
  --seed 42
```

Use `--weights none` for an offline run without downloading ImageNet weights.
The output directory contains `best_model.keras`, `class_indices.json`, and
`training_summary.json`. The split, augmentation seed, and configuration are
recorded in the summary.

## Evaluate and run inference

```bash
python evaluate.py \
  --data-dir data/ultrasound \
  --model artifacts/image/best_model.keras

python predict.py \
  --model artifacts/image/best_model.keras \
  --image path/to/image.jpg \
  --class-indices artifacts/image/class_indices.json
```

## Train the tabular baseline

The CSV must contain a binary target column. Values that contain numbers are
converted to numeric values; missing values are imputed using training data
only.

```bash
python tabular.py \
  --data path/to/clinical.csv \
  --target-column PCOS \
  --output artifacts/clinical_model.joblib \
  --seed 42
```

The command prints accuracy and ROC-AUC for its held-out split. These are
local run results, not repository-wide claims.

## Streamlit app

The app expects model artifacts that are not included in this repository:

```bash
streamlit run app.py
```

If those files are absent, the app displays its configured demo/fallback
behavior. The app's clinical and image model formats must match its
`load_models()` implementation.

## Tests and continuous integration

Run the dataset and utility tests locally:

```bash
python -m pytest -q
```

GitHub Actions runs the same test command after installing `requirements.txt`.

## Limitations

The repository does not include a dataset, trained weights, a data license, or
validated clinical performance. Reproducibility therefore depends on obtaining
an appropriately licensed dataset and recording the exact data version and
training environment used for a run.
