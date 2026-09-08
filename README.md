# 🌸 PCOS Predict — Multi-Modal AI Screening System

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pcos-prediction-system21.streamlit.app/)

**PCOS Predict** is a research and educational AI screening application that combines clinical/lifestyle features, symptom information, and ultrasound image analysis to estimate PCOS risk.

> ⚠️ This project is for research and educational purposes only. It is not a medical diagnostic tool.

## What the system does

- 🩺 Processes clinical and lifestyle features
- 📋 Incorporates symptom-based information
- 🩻 Classifies ultrasound images with EfficientNetB0
- 🔍 Uses Grad-CAM to visualize image regions influencing CNN predictions
- 🧠 Combines available signals into a probability-based risk assessment
- 🌐 Provides an interactive Streamlit interface

## Architecture

```text
Clinical / Lifestyle Data ──> Clinical ML Ensemble ──┐
                                                     │
Symptoms ─────────────────> Risk Features ──────────┼──> Fusion ──> Risk Score
                                                     │
Ultrasound Image ─────────> EfficientNetB0 ─────────┘
                                      │
                                      └──> Grad-CAM Explanation
```

## Clinical ML pipeline

The tabular pipeline uses:

- XGBoost
- LightGBM
- Random Forest
- Logistic Regression meta-learner
- Stacking ensemble learning
- Optuna hyperparameter optimization
- SelectKBest / mutual-information feature selection
- SMOTE class balancing
- Probability calibration

```text
Clinical Dataset
      ↓
Preprocessing
      ↓
Feature Selection
      ↓
SMOTE
      ↓
XGBoost + LightGBM + Random Forest
      ↓
Stacking Classifier
      ↓
Probability Calibration
      ↓
Risk Prediction
```

## Ultrasound model

The image branch uses transfer learning with **EfficientNetB0** pretrained on ImageNet.

```text
EfficientNetB0
      ↓
Global Average Pooling
      ↓
Batch Normalization
      ↓
Dense(256) + Dropout
      ↓
Dense(128)
      ↓
Softmax → PCOS / Non-PCOS
```

Training includes a frozen-backbone stage followed by fine-tuning, image augmentation, mixed precision, and class-weight handling.

## Explainability

Grad-CAM generates heatmaps highlighting image regions that contribute to the CNN prediction. These visualizations are intended to make model behavior easier to inspect rather than treating the prediction as an unexplained score.

## Results

| Component | Metric | Reported result |
|---|---|---:|
| Clinical ensemble | ROC-AUC | 0.93 |
| Clinical ensemble | Recall | ≥88% |
| Clinical ensemble | Average Precision | 0.91 |
| EfficientNetB0 | Validation accuracy | ~87% |

These are project evaluation results and should not be interpreted as clinical performance.

## Project structure

```text
PCOS-Detection-using-Machine-Learning/
├── app.py
├── tabular.py
├── pcos.py
├── test_cnn.py
├── requirements.txt
├── models/
│   ├── pcos_clinical_model.pkl
│   ├── pcos_cnn_model.keras
│   └── weights.weights.h5
└── README.md
```

## Run locally

```bash
git clone https://github.com/pallavi12-code/PCOS-Detection-using-Machine-Learning.git
cd PCOS-Detection-using-Machine-Learning
pip install -r requirements.txt
streamlit run app.py
```

For training:

```bash
python tabular.py
python pcos.py
python test_cnn.py
```

A hosted demo is available at: https://pcos-prediction-system21.streamlit.app/

## Tech stack

**Python · Scikit-learn · XGBoost · LightGBM · Optuna · TensorFlow/Keras · EfficientNetB0 · Grad-CAM · Streamlit**

## Future work

- Evaluate on larger and more diverse datasets
- Add stronger external validation
- Improve model monitoring and reproducibility
- Explore mobile/edge deployment
- Add more rigorous uncertainty analysis

## Author

**Pallavi Reddy**  
B.E. Artificial Intelligence & Machine Learning, CBIT Hyderabad
