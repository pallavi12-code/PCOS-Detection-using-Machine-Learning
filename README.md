# 🌸 PCOS Predict — Multi-Modal AI Detection System

An AI-powered PCOS (Polycystic Ovary Syndrome) screening system that combines **Machine Learning, Deep Learning, Computer Vision, and Explainable AI** to predict PCOS risk using clinical data, symptoms, and ultrasound images.

The system provides a probability-based risk assessment through a Streamlit web application by combining:

- 🩺 Clinical & lifestyle data analysis
- 📋 Symptom-based risk scoring
- 🧠 Ultrasound image classification using EfficientNetB0
- 🔍 GradCAM-based explainability


## 📌 Problem Statement

Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age. Early detection is difficult because symptoms vary among individuals and may overlap with other conditions.

This project develops an end-to-end AI screening system that integrates multiple data sources:

- Symptom information
- Clinical laboratory parameters
- Ultrasound images

to generate a PCOS risk prediction score.

> ⚠️ This project is intended for research and educational purposes only and does not replace medical diagnosis.


# 🏗️ System Architecture

            User Input

                |
 ---------------------------------
 |               |               |
 ↓               ↓               ↓

Symptoms Clinical ML Ultrasound CNN
Analysis Ensemble EfficientNetB0

                |
                ↓

        Adaptive Fusion Layer

                |
                ↓

         Final Risk Score

                |
      ---------------------
      |                   |
      ↓                   ↓

Probability Score GradCAM Heatmap



# 🧠 Machine Learning Model

## Clinical Data Prediction

The clinical model predicts PCOS risk using medical and lifestyle features.

### Algorithms Used

- XGBoost
- LightGBM
- Random Forest
- Logistic Regression Meta Learner

### Techniques Applied

- Stacking Ensemble Learning
- Optuna Hyperparameter Optimization
- SelectKBest Feature Selection
- Mutual Information Ranking
- SMOTE Class Balancing
- Probability Calibration


## ML Pipeline


Clinical Dataset

    ↓

Data Preprocessing

    ↓

Feature Selection

    ↓

SMOTE Balancing

    ↓

XGBoost + LightGBM + Random Forest

    ↓

Stacking Classifier

    ↓

Probability Calibration

    ↓

PCOS Prediction



# 🩻 Deep Learning Model

## Ultrasound Image Classification

The ultrasound classification module uses transfer learning with EfficientNetB0.

### CNN Architecture


EfficientNetB0
(Pretrained ImageNet)

    ↓

Global Average Pooling

    ↓

Batch Normalization

    ↓

Dense Layer (256)

    ↓

Dropout (0.5)

    ↓

Dense Layer (128)

    ↓

Softmax Output

    ↓

PCOS / Non-PCOS Classification



## Training Strategy

- Two-phase transfer learning
- Frozen base model training
- Fine-tuning last 100 layers
- Mixed precision training
- Class weight handling


## Image Augmentation

- Horizontal Flip
- Random Crop
- Brightness Adjustment
- Contrast Variation
- Saturation Modification


# 🔍 Explainable AI (GradCAM)

GradCAM is used to visualize important regions of ultrasound images that influence CNN predictions.

Benefits:

- Improves model interpretability
- Explains AI decisions
- Provides visual prediction evidence


# 🌐 Streamlit Application

The project includes a healthcare-focused Streamlit web interface.

## Workflow


Personal Information

    ↓

Symptom Checklist

    ↓

Clinical Lab Values

    ↓

Ultrasound Upload

    ↓

AI Prediction

    ↓

Risk Score + Explanation



## Features

✅ Multi-modal AI prediction  
✅ Adaptive fusion of available inputs  
✅ Clinical ML prediction  
✅ Ultrasound CNN classification  
✅ GradCAM visualization  
✅ Probability-based risk score  
✅ Interactive healthcare UI  


# 📊 Results

| Model | Metric | Performance |
|------|--------|-------------|
| Clinical Ensemble | ROC-AUC | 0.93 |
| Clinical Ensemble | Recall | ≥88% |
| Clinical Ensemble | Average Precision | 0.91 |
| EfficientNetB0 CNN | Validation Accuracy | ~87% |


# 📂 Project Structure


pcos-predict/

│
├── app.py
│ └── Streamlit Application
│
├── tabular.py
│ └── Clinical ML Model
│
├── pcos.py
│ └── CNN Training Model
│
├── test_cnn.py
│ └── CNN Prediction
│
├── requirements.txt
│
├── models/
│ ├── pcos_clinical_model.pkl
│ ├── pcos_cnn_model.keras
│ └── weights.weights.h5
│
└── README.md



# ⚙️ Tech Stack

## Programming Language

- Python


## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Optuna
- SMOTE


## Deep Learning

- TensorFlow
- Keras
- EfficientNetB0


## Explainable AI

- GradCAM
- SHAP


## Deployment

- Streamlit


# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/pcos-predict.git

cd pcos-predict
Install Dependencies
pip install -r requirements.txt
Add Model Files

Place trained models inside:

models/

├── pcos_clinical_model.pkl
├── pcos_cnn_model.keras
└── weights.weights.h5
▶️ Run Application
streamlit run app.py
🏋️ Training
Train Clinical Model
python tabular.py
Train CNN Model
python pcos.py
Run CNN Prediction
python test_cnn.py
⭐ Key Highlights
Multi-modal healthcare AI system
Machine Learning + Deep Learning integration
Ensemble classification model
EfficientNetB0 medical image classifier
Explainable AI using GradCAM
High recall optimized screening
Probability calibrated predictions
Streamlit deployment
🔮 Future Enhancements
Mobile application deployment
Hospital database integration
Larger medical datasets
Advanced transformer-based medical models
Real-time patient monitoring
⚕️ Medical Disclaimer

This system is developed for research and educational purposes only.

The prediction should not be considered a medical diagnosis.

PCOS diagnosis must be confirmed by qualified healthcare professionals using clinical examination, laboratory tests, medical history, and ultrasound evaluation.

👩‍💻 Author

Rinki (Marikanti Pallavi Reddy)

B.E. Artificial Intelligence & Machine Learning
Chaitanya Bharathi Institute of Technology, Hyderabad





