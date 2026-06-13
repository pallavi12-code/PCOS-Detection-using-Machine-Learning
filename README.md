# PCOS-Detection-using-Machine-Learning
Developed a Machine Learning-based PCOS detection system using medical and lifestyle data. Performed data preprocessing, feature analysis, and classification modeling to predict PCOS risk. Implemented ML algorithms using Python, Pandas, and Scikit-learn for early healthcare prediction and analysis.
🌸 PCOS Predict — Multi-Modal AI Detection System


A clinical AI screening tool combining symptom analysis, tabular clinical data (ML ensemble), and ultrasound image classification (EfficientNetB0 CNN) — fused into a Streamlit web app with GradCAM explainability.






📌 Problem Statement

Polycystic Ovary Syndrome (PCOS) affects 1 in 10 women of reproductive age, yet diagnosis is often delayed due to symptom overlap, inconsistent lab thresholds, and limited access to specialists. This project builds an end-to-end AI screening system that integrates three modalities — symptom checklist, clinical lab values, and ovarian ultrasound images — to deliver a probability-weighted risk assessment.


🧩 System Architecture

┌─────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐
│  Symptom Score  │   │  Clinical ML Ensemble     │   │  CNN (EfficientNet) │
│  (Weighted      │   │  XGBoost + LightGBM + RF  │   │  Ultrasound Image   │
│  checklist)     │   │  Stacking + Calibration   │   │  Classification     │
└────────┬────────┘   └────────────┬─────────────┘   └──────────┬──────────┘
         │                         │                              │
         └────────────────┬────────┘                             │
                          │      Adaptive Fusion Weights          │
                          └──────────────┬───────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   Final Risk Score   │
                              │   + GradCAM Overlay  │
                              └─────────────────────┘


🧠 Models

1. Clinical ML Ensemble (tabular.py)

ComponentDetailBase ModelsXGBoost, LightGBM, Random ForestMeta-learnerLogistic Regression (C=0.1)MethodStacking + Isotonic/Sigmoid CalibrationTuningOptuna (50 trials, AUC objective)Imbalance handlingPartial SMOTE (0.75) + scale_pos_weightFeature selectionSelectKBest (Mutual Information, k=30)Recall target≥ 88% at optimal threshold

Key design choices:


SMOTE applied inside CV folds (no data leakage from v4)
Stronger regularization (min_child_weight, num_leaves) to close the CV→Test AUC gap
Threshold tuned to maximize PCOS recall (clinical priority: minimize missed diagnoses)


2. CNN — Ultrasound Image Classifier (pcos.py)

ComponentDetailBase ModelEfficientNetB0 (pretrained ImageNet)Custom HeadGAP → BN → Dense(256) → Dropout(0.5) → Dense(128) → SoftmaxTrainingPhase 1: frozen base (10 epochs) → Phase 2: fine-tune last 100 layersPrecisionMixed Float16 for faster trainingAugmentationFlip, brightness, contrast, saturation, random cropClass handlingComputed class weights (sklearn)CallbacksEarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best AUC)

3. Streamlit App (app.py)


4-step form: Personal info → Symptom checklist → Clinical lab values → Ultrasound image upload
Adaptive fusion: weights between the three modalities shift based on data availability and agreement
GradCAM explainability: highlights regions of the ultrasound the CNN focused on
Risk gauge chart: semicircular probability visualization
Medical disclaimer and adjustable fusion threshold via sidebar slider



📊 Results

ModelMetricValueClinical EnsembleTest AUC (ROC)~0.93Clinical EnsemblePCOS Recall≥ 88%Clinical EnsembleAverage Precision~0.91CNN (EfficientNetB0)Validation Accuracy~87%Fusion SystemModalities supported1–3 (adaptive)


🗂️ Project Structure

pcos-predict/
├── app.py                  # Streamlit multi-modal web app
├── pcos.py                 # CNN training (EfficientNetB0)
├── tabular.py              # Clinical ML ensemble (XGB + LGB + RF stacking)
├── test_cnn.py             # Standalone CNN inference script
├── requirements.txt
├── models/                 # Saved model files (not tracked in git)
│   ├── pcos_clinical_model.pkl
│   ├── weights.weights.h5
│   └── pcos_cnn_model.keras
└── README.md


⚙️ Tech Stack

Python TensorFlow 2.13 Keras EfficientNetB0 XGBoost LightGBM Scikit-learn Optuna SMOTE Streamlit Matplotlib SHAP GradCAM Joblib


🚀 How to Run

1. Clone & install

bashgit clone https://github.com/YOUR_USERNAME/pcos-predict.git
cd pcos-predict
pip install -r requirements.txt

2. Add model files

Place your trained model files under models/:

models/pcos_clinical_model.pkl    ← from tabular.py output
models/pcos_cnn_model.keras       ← from pcos.py output

3. Launch the app

bashstreamlit run app.py

4. Train models from scratch (optional)

Clinical model:

bashpython tabular.py   # requires combined_pcos_data.csv

CNN:

bashpython pcos.py      # requires PCOS ultrasound dataset in Google Drive

Inference only:

bashpython test_cnn.py  # loads weights.weights.h5, predicts on test_image.jpg


🔍 Key Features


✅ Three-modality fusion — symptom checklist + clinical labs + ultrasound image
✅ Adaptive weighting — fusion weights adjust automatically based on modalities provided
✅ EfficientNetB0 fine-tuning — two-phase training (frozen → partial unfreeze)
✅ Optuna hyperparameter optimization — 50 trials each for XGBoost and LightGBM
✅ No data leakage — SMOTE applied inside CV folds, preprocessing fit on train only
✅ Recall-optimized threshold — clinically prioritizes catching PCOS over reducing false alarms
✅ GradCAM explainability — visual explanation of CNN decisions on ultrasound images
✅ Calibrated probabilities — isotonic/sigmoid calibration for reliable confidence scores
✅ Streamlit UI with medical UX — clean pastel theme, step-by-step form, risk gauge chart
✅ Demo mode — app runs with simulated scores if model files are absent



⚕️ Medical Disclaimer

This tool is designed for screening and research purposes only. It does not constitute a medical diagnosis. All results should be reviewed by a qualified gynaecologist or endocrinologist. PCOS diagnosis requires clinical examination, blood tests, and imaging by a licensed healthcare professional.


👩‍💻 Author

Rinki (Marikanti Pallavi Reddy)

B.E. AI & ML — Chaitanya Bharathi Institute of Technology, Hyderabad













![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
