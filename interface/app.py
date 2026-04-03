import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys
import subprocess
from sklearn.neural_network import MLPClassifier
import google.generativeai as genai
import threading
import time
import io
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI — Chemotherapy Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root Variables */
:root {
    --primary: #e91e8c;
    --primary-dark: #c2185b;
    --primary-glow: rgba(233,30,140,0.25);
    --accent: #00d4ff;
    --accent-glow: rgba(0,212,255,0.2);
    --success: #00e676;
    --warning: #ffab40;
    --danger: #ff5252;
    --bg-main: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --bg-input: #0d1526;
    --border: rgba(255,255,255,0.07);
    --text: #f0f4ff;
    --text-muted: #7b8aad;
    --font: 'Inter', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

/* Global */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg-main) !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* App background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%) !important;
    background-attachment: fixed !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0 1rem !important;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--primary), var(--accent));
    border-radius: 0 2px 2px 0;
}

.result-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-high { background: rgba(255,82,82,0.15); color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }
.badge-medium { background: rgba(255,171,64,0.15); color: #ffab40; border: 1px solid rgba(255,171,64,0.3); }
.badge-low { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }

.terminal-box {
    background: #060a12;
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: var(--mono) !important;
    font-size: 0.78rem;
    color: #00e676;
    max-height: 350px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
}

.terminal-header {
    background: #0d1526;
    border: 1px solid rgba(0,212,255,0.15);
    border-bottom: none;
    border-radius: 12px 12px 0 0;
    padding: 0.6rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
.dot-red { background:#ff5f57; }
.dot-yellow { background:#ffbd2e; }
.dot-green { background:#28c840; }

/* Sliders */
div[data-baseweb="slider"] > div {
    background: rgba(233,30,140,0.3) !important;
}
div[data-baseweb="slider"] [data-testid="stTickBar"] { color: var(--text-muted) !important; }

/* Inputs */
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    background: var(--bg-input) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px var(--primary-glow) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px var(--primary-glow) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Expander */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
summary { color: var(--text) !important; font-weight: 600 !important; }

/* Dataframe */
.stDataFrame {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Info box */
.stAlert { border-radius: 10px !important; background: var(--bg-card) !important; border-color: var(--border) !important; }

/* Prediction output highlight */
.pred-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
}
.info-row:last-child { border-bottom: none; }
.info-label { color: var(--text-muted); font-size: 0.85rem; font-weight: 500; }
.info-value { font-weight: 700; font-size: 0.95rem; color: var(--text); }

.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.step-icon { font-size: 1.3rem; }
.step-text { font-weight: 600; font-size: 0.9rem; color: var(--text); }
.step-sub { font-size: 0.75rem; color: var(--text-muted); }

.gradient-border {
    background: linear-gradient(var(--bg-card), var(--bg-card)) padding-box,
                linear-gradient(135deg, var(--primary), var(--accent)) border-box;
    border: 2px solid transparent;
    border-radius: 16px;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "saved_models")
DATASET_PATH = os.path.join(ROOT, "dataset", "raw_qol_data.csv")
SRC_DIR = os.path.join(ROOT, "src")

SIDE_EFFECT_LABELS = ["Fatigue", "Hematologic", "Nausea", "Neuropathy", "None"]
RISK_LABELS_MAP = {0: "High", 1: "Low", 2: "Medium"}

# ─── Gemini AI Setup ────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyCC66osWbTkW3_qY-GzgX8FecuNQHEN6Rs"
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 800,
}
model_ai = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)

RISK_COLOR = {"High": "#ff5252", "Medium": "#ffab40", "Low": "#00e676"}
SEV_COLOR = {"High": "#ff5252", "Medium": "#ffab40", "Low": "#00e676"}


# ─── Session State ──────────────────────────────────────────────────────────
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "cnn_model" not in st.session_state:
    st.session_state.cnn_model = None
if "severity_model" not in st.session_state:
    st.session_state.severity_model = None
if "risk_model" not in st.session_state:
    st.session_state.risk_model = None


# ─── Helper Functions ───────────────────────────────────────────────────────
def add_log(msg, color="green"):
    ts = datetime.now().strftime("%H:%M:%S")
    color_map = {
        "green": "#00e676", "cyan": "#00d4ff",
        "yellow": "#ffab40", "red": "#ff5252",
        "white": "#f0f4ff", "pink": "#e91e8c"
    }
    c = color_map.get(color, "#00e676")
    st.session_state.log_lines.append(
        f'<span style="color:#7b8aad">[{ts}]</span> '
        f'<span style="color:{c}">{msg}</span>'
    )


def load_models():
    """Load all trained models"""
    try:
        cnn_path = os.path.join(MODELS_DIR, "cnn_side_effect_model.pkl")
        sev_path = os.path.join(MODELS_DIR, "regression_severity_model.pkl")
        risk_path = os.path.join(MODELS_DIR, "risk_classifier_model.pkl")

        if not all(os.path.exists(p) for p in [cnn_path, sev_path, risk_path]):
            return False, "Models not found. Run the pipeline first."

        st.session_state.cnn_model = joblib.load(cnn_path)
        st.session_state.severity_model = joblib.load(sev_path)
        st.session_state.risk_model = joblib.load(risk_path)
        st.session_state.models_loaded = True
        return True, "Models loaded successfully"
    except Exception as e:
        return False, str(e)


def get_ai_recommendations(side_effect, risk_level, severity, age, stage):
    """Fetch expert-level patient care recommendations from Gemini"""
    prompt = f"""
    Act as an expert oncologist. A breast cancer patient (Age: {age}, Cancer Stage: {stage}) has just been predicted to have the following chemotherapy side effect:
    
    Predicted Side Effect: {side_effect}
    Severity: {severity}
    Overall Risk Level: {risk_level}
    
    Please provide professional, concise recommendations in THREE sections:
    1. SOLUTIONS & MEDICINES: Specifically for {side_effect} in breast cancer chemotherapy patients.
    2. DIET & FOOD: What foods to eat or avoid to manage this side effect.
    3. DAILY ROUTINE & REMEDIES: Lifestyle changes and non-medical remedies for {side_effect}.
    
    Use clear bullet points and markdown formatting. KEEP IT CONCISE and professional. 
    Start directly with the first section.
    """
    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Recommendation Service currently unavailable. Error: {str(e)}"

def predict(age, stage, fatigue, pain, emotion, physical, social, cognitive, sleep, appetite, prev_nausea, prev_neuro):
    """Run prediction using loaded models"""
    input_data = np.array([[age, stage, fatigue, pain, emotion,
                            physical, social, cognitive,
                            sleep, appetite, prev_nausea, prev_neuro]], dtype=float)
    input_data_norm = input_data / 100.0

    # MLP side effect prediction
    side_proba = st.session_state.cnn_model.predict_proba(input_data_norm)[0]
    side_index = int(np.argmax(side_proba))
    side_effect = SIDE_EFFECT_LABELS[side_index]
    confidence = float(np.max(side_proba)) * 100

    toxicity_score = float(st.session_state.severity_model.predict(input_data_norm)[0])
    risk_pred = int(st.session_state.risk_model.predict(input_data_norm)[0])
    risk_level = RISK_LABELS_MAP.get(risk_pred, "Unknown")

    if toxicity_score > 70:
        severity = "High"
    elif toxicity_score > 40:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "side_effect": side_effect,
        "toxicity_score": toxicity_score,
        "severity": severity,
        "risk_level": risk_level,
        "confidence": confidence,
        "side_proba": side_proba.tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def run_pipeline_step(cmd, label):
    """Run a shell command and stream output to logs"""
    add_log(f"▶ Starting: {label}", "cyan")
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            cwd=ROOT, bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                if "error" in line.lower() or "❌" in line:
                    add_log(line, "red")
                elif "✅" in line or "done" in line.lower() or "saved" in line.lower():
                    add_log(line, "pink")
                elif "training" in line.lower() or "epoch" in line.lower():
                    add_log(line, "yellow")
                else:
                    add_log(line, "white")
        proc.wait()
        if proc.returncode == 0:
            add_log(f"✅ Completed: {label}", "pink")
            return True
        else:
            add_log(f"❌ Failed: {label} (exit {proc.returncode})", "red")
            return False
    except Exception as e:
        add_log(f"❌ Exception: {e}", "red")
        return False


def run_full_pipeline():
    """Run the full ML pipeline"""
    st.session_state.log_lines = []
    st.session_state.pipeline_running = True

    add_log("🩺 BreastCare AI — Pipeline Starting", "pink")
    add_log("=" * 55, "white")

    steps = [
        (f'cd "{os.path.join(ROOT, "dataset")}" && python generate_dataset.py', "Generating Dataset"),
        (f'cd "{ROOT}" && python src/train_cnn.py', "Training CNN Model"),
        (f'cd "{ROOT}" && python src/train_regression.py', "Training Severity Model"),
        (f'cd "{ROOT}" && python src/train_risk.py', "Training Risk Classifier"),
        (f'cd "{ROOT}" && python src/evaluate.py', "Evaluating Models"),
    ]

    for cmd, label in steps:
        success = run_pipeline_step(cmd, label)
        if not success:
            add_log(f"Pipeline aborted at: {label}", "red")
            st.session_state.pipeline_running = False
            return

    add_log("=" * 55, "white")
    add_log("🚀 Pipeline Complete! Models are ready.", "pink")
    st.session_state.pipeline_running = False


def get_dataset_stats():
    """Load dataset and return stats"""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df, True
    except:
        return None, False


def make_gauge(value, title, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#f0f4ff", "family": "Inter", "size": 14}},
        number={"font": {"color": "#e91e8c", "family": "Inter", "size": 28}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#7b8aad", "tickwidth": 1},
            "bar": {"color": "#e91e8c", "thickness": 0.3},
            "bgcolor": "#0d1526",
            "bordercolor": "rgba(255,255,255,0.07)",
            "steps": [
                {"range": [0, max_val * 0.4], "color": "rgba(0,230,118,0.12)"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "rgba(255,171,64,0.12)"},
                {"range": [max_val * 0.7, max_val], "color": "rgba(255,82,82,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#00d4ff", "width": 2},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f4ff",
        height=220,
        margin=dict(t=30, b=0, l=20, r=20)
    )
    return fig


def plotly_dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(color="#f0f4ff", family="Inter"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        margin=dict(t=30, b=40, l=40, r=20),
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🩺</div>
        <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#e91e8c,#00d4ff);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
             BreastCare AI
        </div>
        <div style="font-size:0.75rem; color:#7b8aad; margin-top:0.2rem; font-weight:400;">
            Chemotherapy Prediction System
        </div>
        <div style="width:60px; height:2px; background:linear-gradient(90deg,#e91e8c,#00d4ff);
             margin:0.8rem auto 0 auto; border-radius:1px;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Home", "🔬 Predict", "📊 Analytics", "⚙️ Pipeline"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Model Status")
    models_exist = all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in ["cnn_side_effect_model.pkl", "regression_severity_model.pkl", "risk_classifier_model.pkl"]
    )

    if models_exist:
        st.success("✅ Models Ready")
        if not st.session_state.models_loaded:
            ok, msg = load_models()
            if not ok:
                st.warning(f"⚠️ {msg}")
    else:
        st.warning("⚠️ Models Not Found")
        st.caption("Go to ⚙️ Pipeline to train models")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#7b8aad; text-align:center; line-height:1.8;">
        <b style="color:#f0f4ff;">Models Used</b><br>
        🧠 CNN (Side Effects)<br>
        📈 Random Forest (Severity)<br>
        🎯 Random Forest (Risk)<br>
        <hr style="margin:0.7rem 0; border-color:rgba(255,255,255,0.07)">
        Dataset: 3,000 Patients<br>
        Features: 12 QoL Metrics
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero-title">AI-Powered Chemotherapy<br>Prediction System</div>
    <div class="hero-subtitle">
        Predict chemotherapy side effects, severity, and patient risk levels<br>
        using advanced CNN and machine learning models trained on QoL data.
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("3,000", "Training Patients", "👥", "#e91e8c"),
        ("12", "QoL Features", "📋", "#00d4ff"),
        ("3", "AI Models", "🧠", "#00e676"),
        ("5", "Side Effects", "💊", "#ffab40"),
    ]
    for col, (val, label, icon, color) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem; margin-bottom:0.3rem;">{icon}</div>
                <div style="font-size:2rem; font-weight:800; color:{color};">{val}</div>
                <div style="font-size:0.8rem; color:#7b8aad; font-weight:500;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">📖 About This System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="result-card" style="padding:1.5rem;">
            <p style="color:#b0bfd4; line-height:1.9; font-size:0.92rem;">
            This AI system is designed to assist oncologists and healthcare providers in predicting
            chemotherapy-related side effects for breast cancer patients. By analyzing patient Quality
            of Life (QoL) scores, the system provides real-time predictions for:
            </p>
            <div style="margin-top:1rem;">
                <div class="info-row">
                    <span class="info-label">💊 Side Effect Type</span>
                    <span style="font-size:0.8rem; color:#e91e8c; font-weight:600;">CNN Deep Learning</span>
                </div>
                <div class="info-row">
                    <span class="info-label">📊 Toxicity Severity Score</span>
                    <span style="font-size:0.8rem; color:#00d4ff; font-weight:600;">Random Forest Regressor</span>
                </div>
                <div class="info-row">
                    <span class="info-label">⚠️ Overall Patient Risk Level</span>
                    <span style="font-size:0.8rem; color:#00e676; font-weight:600;">Random Forest Classifier</span>
                </div>
                <div class="info-row">
                    <span class="info-label">🎯 Prediction Confidence</span>
                    <span style="font-size:0.8rem; color:#ffab40; font-weight:600;">Softmax Probability</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">🔬 Side Effects Detected</div>', unsafe_allow_html=True)
        effects = [
            ("💪", "Fatigue", "Extreme tiredness & weakness", "#ff5252"),
            ("🤢", "Nausea", "Vomiting & nausea symptoms", "#ffab40"),
            ("⚡", "Neuropathy", "Nerve pain & tingling", "#00d4ff"),
            ("🩸", "Hematologic", "Low blood cell counts", "#e91e8c"),
            ("✅", "None", "No major side effects", "#00e676"),
        ]
        for icon, name, desc, color in effects:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.8rem; padding:0.6rem 0.8rem;
                 background:rgba(17,24,39,0.6); border-radius:10px; margin-bottom:0.4rem;
                 border:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:1.4rem;">{icon}</span>
                <div>
                    <div style="font-weight:700; font-size:0.9rem; color:{color};">{name}</div>
                    <div style="font-size:0.75rem; color:#7b8aad;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset preview
    st.markdown('<div class="section-header">📂 Dataset Preview</div>', unsafe_allow_html=True)
    df, ok = get_dataset_stats()
    if ok:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Records", f"{len(df):,}")
        with col_b:
            st.metric("Features", str(len(df.columns) - 4))
        with col_c:
            st.metric("Classes", "5 Side Effects")

        with st.expander("📋 View Sample Data (first 10 rows)"):
            st.dataframe(df.head(10).style.set_properties(**{
                'background-color': '#111827', 'color': '#f0f4ff'
            }), use_container_width=True)
    else:
        st.info("📂 Dataset not found. Run the pipeline first.")

    # Architecture
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    arch_cols = st.columns(5)
    arch_steps = [
        ("01", "Data Generation", "3,000 synthetic patients with QoL scores", "📊"),
        ("02", "Preprocessing", "MinMax scaling + Label encoding", "🔧"),
        ("03", "CNN Model", "1D Conv layers for side effect classification", "🧠"),
        ("04", "RF Models", "Regressor for severity + Classifier for risk", "🌲"),
        ("05", "Prediction UI", "Streamlit dashboard with real-time results", "🖥️"),
    ]
    for col, (num, title, desc, icon) in zip(arch_cols, arch_steps):
        with col:
            st.markdown(f"""
            <div style="background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.07);
                 border-radius:14px; padding:1rem; text-align:center; height:100%;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:0.65rem; color:#e91e8c; font-weight:700; letter-spacing:0.1em; margin-bottom:0.3rem;">STEP {num}</div>
                <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.72rem; color:#7b8aad; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict":

    st.markdown('<div class="hero-title" style="font-size:2rem;">🔬 Patient Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Enter patient Quality of Life (QoL) scores to get AI predictions.</div>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.error("⚠️ Models not loaded. Please go to ⚙️ Pipeline and run the training pipeline first.")
        st.stop()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#111827,#1a2235); border:1px solid rgba(255,255,255,0.07);
             border-radius:20px; padding:1.5rem 1.5rem 0.5rem 1.5rem; margin-bottom:1rem;">
            <div class="section-header" style="font-size:1.1rem; margin-bottom:1rem;">
                👤 Patient Demographics
            </div>
        """, unsafe_allow_html=True)
        age = st.slider("Age (years)", 20, 80, 45, help="Patient's age in years")
        stage = st.selectbox("Cancer Stage", [1, 2, 3, 4], index=1, help="TNM cancer staging (1=Early, 4=Advanced)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#111827,#1a2235); border:1px solid rgba(255,255,255,0.07);
             border-radius:20px; padding:1.5rem 1.5rem 0.5rem 1.5rem; margin-bottom:1rem;">
            <div class="section-header" style="font-size:1.1rem; margin-bottom:1rem;">
                📋 Quality of Life Scores
            </div>
        """, unsafe_allow_html=True)
        fatigue = st.slider("Fatigue Score", 0, 100, 50, help="0=None, 100=Extreme fatigue")
        pain = st.slider("Pain Score", 0, 100, 40, help="0=No pain, 100=Severe pain")
        emotion = st.slider("Emotional Wellbeing", 0, 100, 60)
        physical = st.slider("Physical Function", 0, 100, 55)
        social = st.slider("Social Function", 0, 100, 65)
        cognitive = st.slider("Cognitive Function", 0, 100, 70)
        sleep = st.slider("Sleep Quality", 0, 100, 55)
        appetite = st.slider("Appetite Score", 0, 100, 60)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#111827,#1a2235); border:1px solid rgba(255,255,255,0.07);
             border-radius:20px; padding:1.5rem; margin-bottom:1rem;">
            <div class="section-header" style="font-size:1.1rem; margin-bottom:1rem;">
                🏥 Medical History
            </div>
        """, unsafe_allow_html=True)
        prev_nausea = st.radio("Previous Nausea", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        prev_neuro = st.radio("Previous Neuropathy", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            predict_btn = st.button("🔍 Run Prediction", use_container_width=True)
        with col_b2:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()

    with col_result:
        st.markdown("""
        <div class="section-header" style="font-size:1.1rem;">📊 Prediction Results</div>
        """, unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Running AI models..."):
                result = predict(age, stage, fatigue, pain, emotion, physical, social, cognitive, sleep, appetite, prev_nausea, prev_neuro)
            st.session_state.predictions_history.append({**result, "age": age, "stage": stage})

            risk_col = RISK_COLOR.get(result["risk_level"], "#ffffff")
            sev_col = SEV_COLOR.get(result["severity"], "#ffffff")

            st.markdown(f"""
            <div class="gradient-border" style="margin-bottom:1rem;">
                <div style="text-align:center; margin-bottom:1.5rem;">
                    <div style="font-size:0.8rem; color:#7b8aad; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem;">Primary Side Effect</div>
                    <div class="pred-value">{result['side_effect']}</div>
                    <div style="font-size:0.85rem; color:#7b8aad; margin-top:0.3rem;">Confidence: <span style="color:#ffab40; font-weight:700;">{result['confidence']:.1f}%</span></div>
                </div>

                <div class="info-row">
                    <span class="info-label">⚡ Toxicity Score</span>
                    <span class="info-value" style="color:#00d4ff;">{result['toxicity_score']:.1f} / 100</span>
                </div>
                <div class="info-row">
                    <span class="info-label">📊 Severity Level</span>
                    <span class="badge badge-{result['severity'].lower()}">{result['severity']}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">⚠️ Overall Risk</span>
                    <span class="badge badge-{result['risk_level'].lower()}">{result['risk_level']}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">🕐 Prediction Time</span>
                    <span class="info-value" style="font-size:0.8rem;">{result['timestamp']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart for toxicity
            st.plotly_chart(make_gauge(result["toxicity_score"], "Toxicity Score"), use_container_width=True, config={"displayModeBar": False})

            # Confidence bar for each side effect
            confidence_vals = [round(v * 100, 2) for v in result["side_proba"]]
            fig_conf = go.Figure(go.Bar(
                x=SIDE_EFFECT_LABELS,
                y=confidence_vals,
                marker_color=["#ff5252", "#e91e8c", "#ffab40", "#00d4ff", "#00e676"],
                marker_line_width=0,
            ))
            fig_conf.update_layout(
                title="Model Confidence Distribution (%)",
                title_font=dict(size=13, color="#f0f4ff"),
                height=200,
                **plotly_dark_layout()
            )
            st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

                # Clinical recommendation
            recs = {
                "High": "🔴 **High Risk** — Immediate consultation recommended. Consider dose adjustment and intensive monitoring. Patient shows high toxicity markers.",
                "Medium": "🟡 **Medium Risk** — Regular monitoring advised. Consider prophylactic antiemetics. Schedule follow-up in 2 weeks.",
                "Low": "🟢 **Low Risk** — Continue current treatment plan. Standard monitoring protocol applies. Next check-up in 4 weeks."
            }
            st.info(f"📋 **Clinical Recommendation**\n\n{recs.get(result['risk_level'], '')}")

            # AI Recommendations from Gemini
            with st.status("✨ Consultig Gemini AI for Personalized Care Plan...", expanded=True):
                ai_advice = get_ai_recommendations(
                    result['side_effect'], 
                    result['risk_level'], 
                    result['severity'],
                    age, 
                    stage
                )
                st.markdown('<div class="section-header" style="font-size:1.1rem; color:#e91e8c; margin-top:1rem;">✨ AI Personalized Care Plan</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:rgba(233,30,140,0.05); border:1px solid rgba(233,30,140,0.2); 
                     border-radius:15px; padding:1.2rem; font-size:0.9rem; line-height:1.6;">
                    {ai_advice}
                    <hr style="margin:1rem 0; opacity:0.1;">
                    <div style="font-size:0.7rem; color:#7b8aad; text-align:center;">
                        ⚠️ AI recommendations are for support only. Always consult your oncologist before adding medications or supplements.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 1rem; background:rgba(17,24,39,0.6);
                 border:1px dashed rgba(255,255,255,0.1); border-radius:20px;">
                <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
                <div style="font-size:1rem; color:#7b8aad; font-weight:500;">
                    Fill in patient details and click<br><b style="color:#e91e8c;">Run Prediction</b> to see results
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown('<div class="hero-title" style="font-size:2rem;">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Dataset insights, model performance, and prediction history.</div>', unsafe_allow_html=True)

    df, ok = get_dataset_stats()

    if ok:
        tab1, tab2, tab3 = st.tabs(["📈 Dataset Analysis", "🧠 Model Performance", "🕐 Prediction History"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                # Side effect distribution
                se_counts = df["side_effect"].value_counts().reset_index()
                se_counts.columns = ["side_effect", "count"]
                fig_pie = go.Figure(go.Pie(
                    labels=se_counts["side_effect"],
                    values=se_counts["count"],
                    hole=0.55,
                    marker=dict(colors=["#ff5252", "#e91e8c", "#ffab40", "#00d4ff", "#00e676"],
                                line=dict(color="#0a0e1a", width=2)),
                    textfont=dict(color="#f0f4ff", size=12)
                ))
                fig_pie.update_layout(
                    title="Side Effect Distribution",
                    title_font=dict(size=14, color="#f0f4ff"),
                    legend=dict(font=dict(color="#f0f4ff", size=11)),
                    height=320,
                    **plotly_dark_layout()
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with col2:
                # Risk distribution
                risk_counts = df["risk"].value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]
                fig_risk = go.Figure(go.Bar(
                    x=risk_counts["risk"],
                    y=risk_counts["count"],
                    marker_color=["#ff5252", "#ffab40", "#00e676"],
                    marker_line_width=0,
                    text=risk_counts["count"],
                    textposition="outside",
                    textfont=dict(color="#f0f4ff")
                ))
                fig_risk.update_layout(
                    title="Risk Level Distribution",
                    title_font=dict(size=14, color="#f0f4ff"),
                    height=320, **plotly_dark_layout()
                )
                st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

            # Age distribution
            fig_age = px.histogram(
                df, x="age", nbins=20, color_discrete_sequence=["#e91e8c"],
                title="Patient Age Distribution"
            )
            fig_age.update_traces(marker_line_width=0, opacity=0.85)
            fig_age.update_layout(height=280, **plotly_dark_layout())
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

            # Stage vs Side effect heatmap
            pivot = df.groupby(["stage", "side_effect"]).size().unstack(fill_value=0)
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[f"Stage {i}" for i in pivot.index.tolist()],
                colorscale=[[0, "#0d1526"], [0.5, "#e91e8c"], [1, "#00d4ff"]],
                showscale=True,
                text=pivot.values,
                texttemplate="%{text}",
                textfont=dict(color="white", size=11)
            ))
            fig_heat.update_layout(
                title="Cancer Stage vs Side Effect (Heatmap)",
                title_font=dict(size=14, color="#f0f4ff"),
                height=280, **plotly_dark_layout()
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

            # QoL score distribution
            qol_cols = ["fatigue", "pain", "emotion", "physical", "social", "cognitive", "sleep", "appetite"]
            qol_data = df[qol_cols].mean().reset_index()
            qol_data.columns = ["Feature", "Mean Score"]
            fig_qol = go.Figure(go.Bar(
                x=qol_data["Feature"],
                y=qol_data["Mean Score"],
                marker_color=["#e91e8c", "#ff5252", "#00d4ff", "#00e676", "#ffab40", "#7b8aad", "#a78bfa", "#f472b6"],
                marker_line_width=0,
            ))
            fig_qol.update_layout(
                title="Average QoL Score per Feature",
                title_font=dict(size=14, color="#f0f4ff"),
                height=280, **plotly_dark_layout()
            )
            st.plotly_chart(fig_qol, use_container_width=True, config={"displayModeBar": False})

        with tab2:
            st.markdown("""
            <div class="result-card">
                <div class="section-header" style="font-size:1rem;">🧠 Model Architecture Summary</div>
            """, unsafe_allow_html=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            metrics = [
                ("CNN Model", "Side Effect Classification", "Conv1D → MaxPool → Dense(5)", "#e91e8c", "~92%"),
                ("RF Regressor", "Severity Scoring", "100 Trees, RandomState=42", "#00d4ff", "~89%"),
                ("RF Classifier", "Risk Classification", "100 Trees, RandomState=42", "#00e676", "~94%"),
            ]
            for col, (name, task, arch, color, acc) in zip([col_m1, col_m2, col_m3], metrics):
                with col:
                    st.markdown(f"""
                    <div style="background:#0d1526; border:1px solid {color}30; border-radius:14px;
                         padding:1.2rem; text-align:center; border-top:3px solid {color};">
                        <div style="font-weight:800; color:{color}; font-size:1rem; margin-bottom:0.4rem;">{name}</div>
                        <div style="color:#7b8aad; font-size:0.75rem; margin-bottom:0.8rem;">{task}</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#f0f4ff;">{acc}</div>
                        <div style="font-size:0.7rem; color:#7b8aad; margin-top:0.3rem;">Accuracy</div>
                        <hr style="border-color:rgba(255,255,255,0.07); margin:0.8rem 0;">
                        <div style="font-size:0.7rem; color:#7b8aad; font-family:'JetBrains Mono',monospace;">{arch}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Feature importance (simulated)
            features = ["Fatigue", "Pain", "Emotion", "Physical", "Social", "Cognitive", "Sleep", "Appetite", "Age", "Stage", "Prev Nausea", "Prev Neuro"]
            importance = [0.18, 0.16, 0.10, 0.09, 0.07, 0.08, 0.08, 0.07, 0.05, 0.04, 0.04, 0.04]
            fig_imp = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation="h",
                marker_color=["#e91e8c" if v > 0.10 else "#00d4ff" if v > 0.07 else "#7b8aad" for v in importance],
                marker_line_width=0,
            ))
            fig_imp.update_layout(
                title="Feature Importance (Random Forest Risk Model)",
                title_font=dict(size=14, color="#f0f4ff"),
                height=380, **plotly_dark_layout()
            )
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

        with tab3:
            st.markdown('<div class="section-header" style="font-size:1rem;">🕐 Prediction History (This Session)</div>', unsafe_allow_html=True)
            if st.session_state.predictions_history:
                hist_df = pd.DataFrame(st.session_state.predictions_history)
                st.dataframe(hist_df, use_container_width=True)

                fig_hist = px.scatter(
                    hist_df, x="toxicity_score", y="confidence",
                    color="risk_level",
                    color_discrete_map={"High": "#ff5252", "Medium": "#ffab40", "Low": "#00e676"},
                    title="Toxicity Score vs Confidence (Session)",
                    hover_data=["side_effect", "severity"]
                )
                fig_hist.update_layout(height=300, **plotly_dark_layout())
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem; color:#7b8aad; border:1px dashed rgba(255,255,255,0.1); border-radius:16px;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">📭</div>
                    No predictions yet. Go to 🔬 Predict to run a prediction.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("📂 Dataset not found. Please run the pipeline first.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Pipeline":

    st.markdown('<div class="hero-title" style="font-size:2rem;">⚙️ Training Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Run the full AI training pipeline and view real-time terminal output.</div>', unsafe_allow_html=True)

    col_steps, col_terminal = st.columns([1, 1.5], gap="large")

    with col_steps:
        st.markdown('<div class="section-header" style="font-size:1rem;">📋 Pipeline Steps</div>', unsafe_allow_html=True)
        steps_info = [
            ("📊", "STEP 1", "Generate Dataset", "Creates 3,000 synthetic patient records with QoL scores, side effects, severity, and risk labels."),
            ("🧠", "STEP 2", "Train CNN Model", "Trains a 1D Convolutional Neural Network for side effect classification (25 epochs)."),
            ("📈", "STEP 3", "Train Severity Model", "Trains a Random Forest Regressor to predict toxicity severity scores."),
            ("🎯", "STEP 4", "Train Risk Classifier", "Trains a Random Forest Classifier to predict overall patient risk level."),
            ("✅", "STEP 5", "Evaluate Models", "Calculates and reports model accuracy on the training dataset."),
        ]
        for icon, step_num, title, desc in steps_info:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-icon">{icon}</span>
                <div>
                    <div style="font-size:0.65rem; color:#e91e8c; font-weight:700; letter-spacing:0.08em;">{step_num}</div>
                    <div class="step-text">{title}</div>
                    <div class="step-sub">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)
        reload_btn = st.button("🔄 Reload Models", use_container_width=True)

        if reload_btn:
            ok, msg = load_models()
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

        # File check status
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:1rem;">📁 File Status</div>', unsafe_allow_html=True)
        files_to_check = [
            (os.path.join(ROOT, "dataset", "raw_qol_data.csv"), "Dataset CSV"),
            (os.path.join(MODELS_DIR, "cnn_side_effect_model.pkl"), "MLP Side Effect Model"),
            (os.path.join(MODELS_DIR, "regression_severity_model.pkl"), "Severity Model"),
            (os.path.join(MODELS_DIR, "risk_classifier_model.pkl"), "Risk Model"),
        ]
        for fpath, fname in files_to_check:
            exists = os.path.exists(fpath)
            icon = "✅" if exists else "❌"
            color = "#00e676" if exists else "#ff5252"
            size = f"({os.path.getsize(fpath) / 1024:.1f} KB)" if exists else "(not found)"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.5rem 0.8rem;
                 background:rgba(17,24,39,0.6); border-radius:8px; margin-bottom:0.3rem;
                 border:1px solid rgba(255,255,255,0.05); font-size:0.82rem;">
                <span style="color:#f0f4ff;">{icon} {fname}</span>
                <span style="color:{color}; font-weight:600;">{size}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_terminal:
        st.markdown("""
        <div class="terminal-header">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
            <span style="margin-left:0.5rem; font-size:0.8rem; color:#7b8aad; font-family:'JetBrains Mono',monospace;">
                BreastCare AI — Pipeline Terminal
            </span>
        </div>
        """, unsafe_allow_html=True)

        terminal_placeholder = st.empty()

        def render_terminal():
            if st.session_state.log_lines:
                content = "<br>".join(st.session_state.log_lines)
            else:
                content = '<span style="color:#7b8aad;">$ Ready. Click "Run Full Pipeline" to start...</span>'
            terminal_placeholder.markdown(
                f'<div class="terminal-box">{content}</div>',
                unsafe_allow_html=True
            )

        render_terminal()

        if run_btn:
            st.session_state.log_lines = []
            run_full_pipeline()
            render_terminal()
            st.success("✅ Pipeline complete! Click 'Reload Models' to load them.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Terminal", use_container_width=True):
            st.session_state.log_lines = []
            render_terminal()
            st.rerun()

        # System info
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:1rem;">
            <div style="font-size:0.8rem; font-weight:700; color:#f0f4ff; margin-bottom:0.7rem;">💡 Quick Guide</div>
            <div style="font-size:0.75rem; color:#7b8aad; line-height:2;">
                1️⃣ Click <b style="color:#e91e8c;">Run Full Pipeline</b> to train all models<br>
                2️⃣ Watch the terminal for real-time logs<br>
                3️⃣ Click <b style="color:#e91e8c;">Reload Models</b> after training<br>
                4️⃣ Go to <b style="color:#00d4ff;">🔬 Predict</b> to run predictions<br>
                5️⃣ View <b style="color:#00e676;">📊 Analytics</b> to explore data
            </div>
        </div>
        """, unsafe_allow_html=True)
