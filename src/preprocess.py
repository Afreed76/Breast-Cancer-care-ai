import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os


def load_data(path):
    """Load and preprocess the QoL dataset"""

    # Load dataset
    df = pd.read_csv(path)

    # Separate input and output
    X = df.drop(
        ["id", "side_effect", "severity", "risk"],
        axis=1,
        errors="ignore"
    )

    y_side = df["side_effect"]
    y_sev = df["severity"]
    y_risk = df["risk"]

    # Convert categorical to numeric
    X = pd.get_dummies(X)

    # Normalize data (0 to 1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    enc1 = LabelEncoder()
    enc2 = LabelEncoder()
    enc3 = LabelEncoder()

    y_side = enc1.fit_transform(y_side)
    y_sev = enc2.fit_transform(y_sev)
    y_risk = enc3.fit_transform(y_risk)

    return X, y_side, y_sev, y_risk, scaler, enc1, enc2, enc3
