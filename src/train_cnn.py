import joblib
from sklearn.neural_network import MLPClassifier
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "dataset", "raw_qol_data.csv")
MODELS_DIR = os.path.join(ROOT, "saved_models")

# Load data
print("📂 Loading dataset...")
X, y_side, _, _, _, enc1, _, _ = load_data(DATASET_PATH)
print(f"  ✅ Loaded {X.shape[0]} samples, {X.shape[1]} features")
print(f"  🏷️  Classes: {list(enc1.classes_)}")

# MLP Classifier (replaces CNN — compatible with all Python versions)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

print("\n🧠 Training Neural Network (Side Effect Classifier)...")
model.fit(X, y_side)

# Save
os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "cnn_side_effect_model.pkl")
joblib.dump(model, out_path)
print(f"\n✅ Model saved → {out_path}")

# Quick accuracy check
acc = model.score(X, y_side)
print(f"   Training Accuracy: {acc * 100:.2f}%")
