import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "dataset", "raw_qol_data.csv")
MODELS_DIR = os.path.join(ROOT, "saved_models")

print("\n" + "=" * 50)
print("  📊 Model Evaluation Report")
print("=" * 50)

# Load data
print("\n📂 Loading dataset...")
X, y_side, y_sev, y_risk, scaler, enc1, enc2, enc3 = load_data(DATASET_PATH)
print(f"  ✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# ─── MLP Side Effect Model ─────────────────────────
print("\n🧠 Evaluating Neural Network (Side Effect Model)...")
mlp_model = joblib.load(os.path.join(MODELS_DIR, "cnn_side_effect_model.pkl"))
mlp_acc = mlp_model.score(X, y_side)
print(f"  ✅ MLP Accuracy    : {mlp_acc * 100:.2f}%")

# ─── Risk Classifier Evaluation ────────────────────
print("\n🎯 Evaluating Risk Classifier...")
risk_model = joblib.load(os.path.join(MODELS_DIR, "risk_classifier_model.pkl"))
risk_acc = risk_model.score(X, y_risk)
print(f"  ✅ Risk Accuracy   : {risk_acc * 100:.2f}%")

# ─── Severity Model Evaluation ─────────────────────
print("\n📈 Evaluating Severity Regression Model...")
sev_model = joblib.load(os.path.join(MODELS_DIR, "regression_severity_model.pkl"))
pred_sev = sev_model.predict(X)

sev_text = enc2.inverse_transform(y_sev)
sev_map = {"Low": 20, "Medium": 60, "High": 90}
y_sev_scores = np.array([sev_map[s] for s in sev_text])

mse = np.mean((pred_sev - y_sev_scores) ** 2)
rmse = np.sqrt(mse)
print(f"  ✅ Severity RMSE   : {rmse:.2f}")

# ─── Summary ───────────────────────────────────────
print("\n" + "─" * 50)
print("  📋 EVALUATION SUMMARY")
print("─" * 50)
print(f"  MLP Side Effect Model   → Accuracy : {mlp_acc * 100:.2f}%")
print(f"  Risk Classifier Model   → Accuracy : {risk_acc * 100:.2f}%")
print(f"  Severity Regression     → RMSE     : {rmse:.2f}")
print("─" * 50)
print("  ✅ All models evaluated successfully.")
print("=" * 50 + "\n")
