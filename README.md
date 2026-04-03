# 🩺 BreastCare AI — Chemotherapy Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-brightgreen?style=for-the-badge&logo=scikit-learn)

**An AI-powered system that predicts chemotherapy side effects, severity, and patient risk levels using Quality of Life (QoL) data.**

</div>

---

## 📋 Overview

BreastCare AI is a machine learning system designed for oncology support. By analyzing 12 patient Quality of Life (QoL) metrics, the system predicts:

| Prediction | Model | Output |
|---|---|---|
| 💊 Side Effect Type | CNN (1D Conv) | Fatigue / Nausea / Neuropathy / Hematologic / None |
| 📊 Toxicity Severity Score | Random Forest Regressor | Score 0–100 |
| ⚠️ Overall Risk Level | Random Forest Classifier | Low / Medium / High |
| 🎯 Prediction Confidence | Softmax | Percentage (%) |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Train + Launch UI)
```bash
python main.py
```

### 3. Or Launch Streamlit Directly (if models already trained)
```bash
streamlit run interface/app.py
```

The app opens at: **http://localhost:8501**

---

## 🏗️ Project Structure

```
M_chemotherapyprediction/
│
├── main.py                    # Run full pipeline + launch UI
├── requirements.txt           # Python dependencies
│
├── dataset/
│   ├── generate_dataset.py    # Generates 3,000 synthetic patient records
│   └── raw_qol_data.csv       # Generated dataset
│
├── src/
│   ├── preprocess.py          # Data loading & normalization
│   ├── train_cnn.py           # CNN model training (side effects)
│   ├── train_regression.py    # Severity regression training
│   ├── train_risk.py          # Risk classifier training
│   └── evaluate.py            # Model evaluation & metrics
│
├── saved_models/
│   ├── cnn_side_effect_model.h5
│   ├── regression_severity_model.pkl
│   └── risk_classifier_model.pkl
│
└── interface/
    ├── app.py                 # ✨ Premium Streamlit UI
    └── ui.py                  # (Legacy Gradio UI)
```

---

## 🧠 Models Used

### 1. CNN (Convolutional Neural Network)
- **Task**: Side effect classification
- **Architecture**: Conv1D(64) → MaxPool → Conv1D(32) → Flatten → Dense(64) → Dense(5, softmax)
- **Input**: 12 QoL features (reshaped to 1D sequence)
- **Output**: 5 side effect classes

### 2. Random Forest Regressor
- **Task**: Toxicity severity score prediction
- **Estimators**: 100 trees
- **Output**: Score from 0–100 (Low < 40, Medium 40–70, High > 70)

### 3. Random Forest Classifier
- **Task**: Overall patient risk classification
- **Estimators**: 100 trees
- **Output**: Low / Medium / High

---

## 📊 Input Features (12 QoL Metrics)

| Feature | Description | Range |
|---|---|---|
| Age | Patient age | 20–80 |
| Stage | Cancer stage | 1–4 |
| Fatigue | Fatigue score | 0–100 |
| Pain | Pain score | 0–100 |
| Emotion | Emotional wellbeing | 0–100 |
| Physical | Physical function | 0–100 |
| Social | Social function | 0–100 |
| Cognitive | Cognitive function | 0–100 |
| Sleep | Sleep quality | 0–100 |
| Appetite | Appetite score | 0–100 |
| Prev Nausea | Previous nausea | 0 or 1 |
| Prev Neuropathy | Previous neuropathy | 0 or 1 |

---

## 🎨 UI Features

- **🏠 Home** — Project overview, stats, dataset preview, architecture
- **🔬 Predict** — Interactive patient prediction with gauges & recommendations  
- **📊 Analytics** — Charts, heatmaps, feature importance, prediction history
- **⚙️ Pipeline** — Run training pipeline with live terminal output

---

## 📦 Requirements

- Python 3.10+
- TensorFlow 2.15
- Streamlit 1.32
- Scikit-learn 1.4
- Pandas, NumPy, Plotly, Joblib

---

## 🏥 Disclaimer

> This system is for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📄 License

MIT License — Free to use and modify.
