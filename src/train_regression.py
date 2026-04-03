import joblib
from sklearn.ensemble import RandomForestRegressor
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "dataset", "raw_qol_data.csv")
MODELS_DIR = os.path.join(ROOT, "saved_models")

# Load data
print("📂 Loading dataset...")
X, _, y_sev, _, _, _, enc2, _ = load_data(DATASET_PATH)

# Convert labels to numeric scores
sev_text = enc2.inverse_transform(y_sev)
sev_map = {"Low": 20, "Medium": 60, "High": 90}
y = [sev_map[s] for s in sev_text]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("📈 Training Severity Regression Model...")
model.fit(X, y)

# Save
os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "regression_severity_model.pkl")
joblib.dump(model, out_path)
print(f"✅ Regression Model Saved → {out_path}")
