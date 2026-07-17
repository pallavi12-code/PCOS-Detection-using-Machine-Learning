
# 🌸 PCOS Predict — Multi-Modal AI Detection System

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pcos-prediction-system21.streamlit.app/)

**🔗 Live Demo:** [pcos-prediction-system21.streamlit.app](https://pcos-prediction-system21.streamlit.app/)

An AI-powered PCOS (Polycystic Ovary Syndrome) screening system that combines **machine learning, deep learning, computer vision, and explainable AI** to estimate PCOS risk from clinical data, symptoms, and ultrasound images.

The system produces a probability-based risk assessment through a Streamlit web app by combining:

- 🩺 Clinical & lifestyle data analysis
- 📋 Symptom-based risk scoring
- 🧠 Ultrasound image classification (EfficientNetB0)
- 🔍 GradCAM-based explainability

---

## 📌 Problem Statement

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age. Early detection is difficult because symptoms vary between individuals and often overlap with other conditions.

This project builds an end-to-end AI screening system that integrates multiple data sources — symptoms, clinical lab parameters, and ultrasound images — into a single PCOS risk prediction score.

> ⚠️ **Disclaimer:** This project is for research and educational purposes only and does not replace medical diagnosis.

---

## 🏗️ System Architecture

```
                          User Input
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  Symptom Analysis    Clinical ML Ensemble   Ultrasound CNN
                                              (EfficientNetB0)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    Adaptive Fusion Layer
                              │
                              ▼
                      Final Risk Score
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        Probability Score            GradCAM Heatmap
```

---

## 🧠 Clinical Data Model

The clinical model predicts PCOS risk from medical and lifestyle features.

**Algorithms used:**
- XGBoost
- LightGBM
- Random Forest
- Logistic Regression (meta-learner)

**Techniques applied:**
- Stacking ensemble learning
- Optuna hyperparameter optimization
- SelectKBest feature selection + mutual information ranking
- SMOTE class balancing
- Probability calibration

**Pipeline:**

```
Clinical Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Selection
      │
      ▼
SMOTE Balancing
      │
      ▼
XGBoost + LightGBM + Random Forest
      │
      ▼
Stacking Classifier
      │
      ▼
Probability Calibration
      │
      ▼
PCOS Prediction
```

---

## 🩻 Deep Learning Model — Ultrasound Classification

The ultrasound module uses transfer learning with EfficientNetB0.

**CNN architecture:**

```
EfficientNetB0 (pretrained on ImageNet)
      │
      ▼
Global Average Pooling
      │
      ▼
Batch Normalization
      │
      ▼
Dense Layer (256) → Dropout (0.5)
      │
      ▼
Dense Layer (128)
      │
      ▼
Softmax Output → PCOS / Non-PCOS
```

**Training strategy:**
- Two-phase transfer learning: frozen base model, then fine-tuning the last 100 layers
- Mixed precision training
- Class weight handling for imbalance

**Image augmentation:** horizontal flip, random crop, brightness/contrast/saturation adjustment

---

## 🔍 Explainable AI (GradCAM)

GradCAM highlights the ultrasound image regions most influential to each CNN prediction — improving interpretability and giving visual evidence behind each classification rather than a black-box score.

---

## 🌐 Streamlit Application

**Workflow:**

```
Personal Information → Symptom Checklist → Clinical Lab Values
        → Ultrasound Upload → AI Prediction → Risk Score + Explanation
```

**Features:**
- ✅ Multi-modal AI prediction with adaptive fusion of whatever inputs are available
- ✅ Clinical ML prediction
- ✅ Ultrasound CNN classification
- ✅ GradCAM visualization
- ✅ Probability-based risk score
- ✅ Interactive healthcare-focused UI

---

## 📊 Results

| Model | Metric | Performance |
|---|---|---|
| Clinical Ensemble | ROC-AUC | 0.93 |
| Clinical Ensemble | Recall | ≥88% |
| Clinical Ensemble | Average Precision | 0.91 |
| EfficientNetB0 CNN | Validation Accuracy | ~87% |

---

## 📂 Project Structure

```
pcos-predict/
├── app.py                    # Streamlit application
├── tabular.py                 # Clinical ML model (training)
├── pcos.py                    # CNN training model
├── test_cnn.py                 # CNN prediction/inference
├── requirements.txt
├── models/
│   ├── pcos_clinical_model.pkl
│   ├── pcos_cnn_model.keras
│   └── weights.weights.h5
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, Optuna, SMOTE |
| Deep Learning | TensorFlow, Keras, EfficientNetB0 |
| Explainable AI | GradCAM, SHAP |
| Deployment | Streamlit |

---

## 🚀 Installation

> 💡 Want to try it without installing anything? Use the [live demo](https://pcos-prediction-system21.streamlit.app/).

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/pcos-predict.git
cd pcos-predict
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add model files**

Place trained models inside `models/`:
```
models/
├── pcos_clinical_model.pkl
├── pcos_cnn_model.keras
└── weights.weights.h5
```

**4. Run the application**
```bash
streamlit run app.py
```

---

## 🏋️ Training

**Train the clinical model:**
```bash
python tabular.py
```

**Train the CNN model:**
```bash
python pcos.py
```

**Run CNN prediction/inference:**
```bash
python test_cnn.py
```

---

## ⭐ Key Highlights

- Multi-modal healthcare AI system combining ML and deep learning
- Ensemble classification with Optuna-tuned stacking
- EfficientNetB0-based medical image classifier
- Explainable AI via GradCAM
- High-recall-optimized screening with probability calibration
- Deployed as an interactive Streamlit app

---

## 🔮 Future Enhancements

- Mobile application deployment
- Hospital database integration
- Larger, more diverse medical datasets
- Advanced transformer-based medical models
- Real-time patient monitoring

---

## ⚕️ Medical Disclaimer

This system is developed for **research and educational purposes only**. Its predictions are not a medical diagnosis. PCOS diagnosis must be confirmed by qualified healthcare professionals using clinical examination, laboratory tests, medical history, and ultrasound evaluation.

---

## 👩‍💻 Author

** (Marikanti Pallavi Reddy)**
B.E. Artificial Intelligence & Machine Learning
Chaitanya Bharathi Institute of Technology, Hyderabad
