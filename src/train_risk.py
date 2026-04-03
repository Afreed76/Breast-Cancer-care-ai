import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "dataset", "raw_qol_data.csv")
MODELS_DIR = os.path.join(ROOT, "saved_models")

# Load data
print("📂 Loading dataset...")
X, _, _, y_risk, _, _, _, _ = load_data(DATASET_PATH)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("🎯 Training Risk Classifier Model...")
model.fit(X, y_risk)

# Save
os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "risk_classifier_model.pkl")
joblib.dump(model, out_path)
print(f"✅ Risk Model Saved → {out_path}")
