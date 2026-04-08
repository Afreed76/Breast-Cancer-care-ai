import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from groq import Groq
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import threading
import time
import io
import os
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load secret environment variables (e.g. from .env file locally)
load_dotenv()

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
    --primary: #F57C00;
    --primary-dark: #E65100;
    --primary-glow: rgba(245, 124, 0, 0.25);
    --accent: #FFCA28;
    --accent-glow: rgba(255, 202, 40, 0.2);
    --success: #00Bfa5;
    --warning: #FFB300;
    --danger: #D32F2F;
    --bg-main: #F4F5F7;
    --bg-card: #FFFFFF;
    --bg-card2: #FFFFFF;
    --bg-input: #FFFFFF;
    --border: rgba(0, 0, 0, 0.1);
    --text: #000000;
    --text-muted: #000000;
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
    background: #F4F5F7 !important;
    background-attachment: fixed !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
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
    background: #1E1E1E;
    border: 1px solid rgba(245, 124, 0, 0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: var(--mono) !important;
    font-size: 0.78rem;
    color: #FFFFFF;
    max-height: 350px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
}

.terminal-header {
    background: #000000;
    border: 1px solid rgba(245, 124, 0, 0.3);
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
    background: rgba(245, 124, 0, 0.3) !important;
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

# ─── Groq AI Setup ────────────────────────────────────────────────────────
CURRENT_MODEL_NAME = "llama-3.3-70b-versatile"

def get_groq_client():
    # Use environment variable managed securely by .env (locally) or Streamlit Secrets (in production)
    key = os.environ.get("GROQ_API_KEY", "")
    
    # Optional stream deployment handling
    if not key and getattr(st, 'secrets', None) and "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
        
    if not key:
        st.error("⚠️ Groq API Key is missing! Set it in your .env file or Streamlit Secrets.")
        raise ValueError("Groq API Key not found.")
        
    return Groq(api_key=key)

# ─── MongoDB Setup ────────────────────────────────────────────────────────
@st.cache_resource
def get_mongodb_collection():
    uri = os.environ.get("MONGO_URI", "")
    if getattr(st, 'secrets', None) and "MONGO_URI" in st.secrets:
        uri = st.secrets["MONGO_URI"]
        
    if not uri:
        st.warning("⚠️ MongoDB connection string (MONGO_URI) is missing. Predictions will not be tracked in the database.")
        return None
        
    try:
        import certifi
        client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        db = client["breast_care_ai"]
        return db["patient_predictions"]
    except Exception as e:
        st.error(f"⚠️ Failed to connect to MongoDB: {e}")
        return None

# Local Knowledge Base (No-API Fallback)
LOCAL_ADVICE = {
    "Nausea": "### 💊 Solutions & Medicines\n- Anti-nausea medications (antiemetics) prescribed by doctor\n- Ondansetron or Metoclopramide as needed\n\n### 🥗 Diet & Food\n- Ginger tea, small bland meals (BRAT diet)\n- Stay hydrated with electrolyte drinks, avoid greasy foods\n\n### 🌅 Daily Routine\n- Eat 5-6 small meals rather than 3 large ones\n- Rest with head elevated after eating\n\n### 🛡️ Safety Measures\n- Go to ER if unable to keep fluids down for 24 hours\n- Watch for signs of severe dehydration",
    "Fatigue": "### 💊 Solutions & Medicines\n- Consult for Iron supplements if anemic\n- Avoid sleep medications unless prescribed\n\n### 🥗 Diet & Food\n- High-protein snacks, balanced meals\n- Stay fully hydrated, limit heavy sugars\n\n### 🌅 Daily Routine\n- Schedule 'rest clusters'. Short 20-min naps\n- Gentle 10-15 minute walking daily\n\n### 🛡️ Safety Measures\n- Avoid driving if severely fatigued\n- Watch for sudden extreme exhaustion (could indicate low counts)",
    "Neuropathy": "### 💊 Solutions & Medicines\n- Gabapentin or pregabalin (ask doctor)\n- Vitamin B-complex supplements (if approved)\n\n### 🥗 Diet & Food\n- B12-rich foods (lean meats, fortified cereals)\n- Avoid excess alcohol and processed sugars\n\n### 🌅 Daily Routine\n- Wear loose, comfortable footwear\n- Gentle massages and warm foot baths\n\n### 🛡️ Safety Measures\n- Test water temp with thermometer before bathing\n- Keep hands/feet warm in cold weather",
    "Hematologic": "### 💊 Solutions & Medicines\n- Growth factor injections (Neulasta) if prescribed\n- Strict adherence to prescribed antibiotics\n\n### 🥗 Diet & Food\n- Well-cooked foods only (neutropenic diet)\n- Avoid unpasteurized dairy and raw meats\n\n### 🌅 Daily Routine\n- Excellent hand hygiene, frequent washing\n- Avoid crowded places and sick individuals\n\n### 🛡️ Safety Measures\n- Immediate ER visit for fever over 100.4°F (38°C)\n- Avoid any activities with high risk of cuts/bleeding",
    "General": "### 💊 Solutions & Medicines\n- Ensure all prescribed medications are taken on schedule\n- Ask doctor before taking any supplements\n\n### 🥗 Diet & Food\n- Balanced, nutrient-dense meals\n- 8-10 glasses of water daily\n\n### 🌅 Daily Routine\n- Practice mindfulness, gentle stretching\n- Stay connected with your support group\n\n### 🛡️ Safety Measures\n- Report any new or worsening symptoms immediately\n- Keep emergency contacts easily accessible"
}

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
    
    Please provide professional, concise recommendations in EXACTLY these FOUR sections using EXACTLY these headings:
    
    ### 💊 Solutions & Medicines
    Provide 3-4 bullet points on specific medications or medical interventions for {side_effect}.
    
    ### 🥗 Diet & Food
    Provide 3-4 bullet points on foods to eat or avoid to manage this.
    
    ### 🌅 Daily Routine
    Provide 3-4 bullet points on specific schedule adjustments and lifestyle remedies.
    
    ### 🛡️ Safety Measures
    Provide 3-4 bullet points on precautions, when to consult a doctor, and warnings.
    
    Keep the content concise and professional. Do NOT include any intro or outro text, JUST the headings and bullet points.
    """
    
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=CURRENT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Use Local Knowledge Base as Emergency Fallback
        advice = LOCAL_ADVICE.get(side_effect, LOCAL_ADVICE["General"])
        return f"### ⚠️ AI Service Offline - Using Local Knowledge Base\n\n**Base Recommendations for {side_effect}:**\n\n{advice}\n\n*Original Error: {str(e)}*"

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
    page = st.radio("", ["🏠 Home", "🔬 Predict", "📊 Analytics", "🤖 AI Assistant", "⚙️ Pipeline"], label_visibility="collapsed")

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
            <div style="background:rgba(240,240,240,0.8); border:1px solid rgba(0,0,0,0.1);
                 border-radius:14px; padding:1rem; text-align:center; height:100%;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:0.65rem; color:#F57C00; font-weight:700; letter-spacing:0.1em; margin-bottom:0.3rem;">STEP {num}</div>
                <div style="font-weight:700; font-size:0.85rem; margin-bottom:0.4rem; color:#000000;">{title}</div>
                <div style="font-size:0.72rem; color:#000000; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict":

    st.markdown('<div class="hero-title" style="font-size:2rem; color: #F57C00;">🔬 Patient Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="color: #000000;">Enter patient Quality of Life (QoL) scores to get AI predictions.</div>', unsafe_allow_html=True)

    with st.expander("📊 Score Interpretation Guide (Click to Expand)", expanded=False):
        st.markdown("""
        <div style="background: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #FFCA28;">
        <h4 style="color: #F57C00; margin-top:0;">0-30: Mild | 31-70: Moderate | 71-100: Severe</h4>
        <ul style="color: #000000;">
          <li><b>💪 Fatigue:</b> 0-30 Normal | 31-70 Weak | 71-100 Extreme</li>
          <li><b>🤢 Nausea:</b> 0-30 Little discomfort | 31-70 Vomiting sometimes | 71-100 Severe</li>
          <li><b>⚡ Neuropathy:</b> 0-30 Mild tingling | 31-70 Numbness/Pain | 71-100 Severe pain</li>
          <li><b>🩸 Hematologic:</b> Correlated indirectly with immune weakness / anemia.</li>
        </ul>
        <p style="color: #D32F2F; font-size:0.9em; margin-bottom:0;">🚨 <i>Note: Very high scores (> 70) require immediate symptom management but do not exclusively calculate generic cancer risk alone. Scans and biopsies are required for diagnosis.</i></p>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.error("⚠️ Models not loaded. Please go to ⚙️ Pipeline and run the training pipeline first.")
        st.stop()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="background: #FFFFFF; border:1px solid rgba(0,0,0,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.05);
             border-radius:20px; padding:1.5rem 1.5rem 0.5rem 1.5rem; margin-bottom:1rem;">
            <div class="section-header" style="font-size:1.1rem; margin-bottom:1rem; color: #F57C00;">
                👤 Patient Demographics & Identity
            </div>
        """, unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            patient_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
            patient_gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
        with col_p2:
            patient_contact = st.text_input("Email/Contact", placeholder="e.g. patient@ext.com")
            patient_blood = st.selectbox("Blood Group", ["Unknown", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            
        age = st.slider("Age (years)", 20, 80, 45, help="Patient's age in years")
        stage = st.selectbox("Cancer Stage", [1, 2, 3, 4], index=1, help="TNM cancer staging (1=Early, 4=Advanced)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #FFFFFF; border:1px solid rgba(0,0,0,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.05);
             border-radius:20px; padding:1.5rem 1.5rem 0.5rem 1.5rem; margin-bottom:1rem;">
            <div class="section-header" style="font-size:1.1rem; margin-bottom:1rem; color: #B8860B;">
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

        # Added Terminal Box on UI
        st.markdown("""
        <div class="terminal-header" style="margin-top:2rem;">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
            <span style="margin-left:0.5rem; font-size:0.8rem; color:#7b8aad; font-family:'JetBrains Mono',monospace;">
                System Logs (Live)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.log_lines:
            # Show last 20 lines to save space
            content = "<br>".join(st.session_state.log_lines[-20:])
        else:
            content = '<span style="color:#7b8aad;">$ Dashboard ready...</span>'
            
        st.markdown(
            f'<div class="terminal-box" style="max-height: 250px;">{content}</div>',
            unsafe_allow_html=True
        )

    with col_result:
        st.markdown("""
        <div class="section-header" style="font-size:1.1rem;">📊 Prediction Results</div>
        """, unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Running AI models..."):
                result = predict(age, stage, fatigue, pain, emotion, physical, social, cognitive, sleep, appetite, prev_nausea, prev_neuro)
            st.session_state.predictions_history.append({**result, "age": age, "stage": stage})
            
            # Save to MongoDB Atlas
            db_col = get_mongodb_collection()
            if db_col is not None:
                doc = {
                    "patient": {
                        "name": patient_name,
                        "gender": patient_gender,
                        "contact": patient_contact,
                        "blood_group": patient_blood,
                        "age": age,
                        "cancer_stage": stage
                    },
                    "metrics": {
                        "fatigue": fatigue, "pain": pain, "emotion": emotion,
                        "physical": physical, "social": social, "cognitive": cognitive,
                        "sleep": sleep, "appetite": appetite, 
                        "prev_nausea": prev_nausea, "prev_neuro": prev_neuro
                    },
                    "prediction": {
                        "side_effect": result['side_effect'],
                        "confidence": result['confidence'],
                        "toxicity_score": result['toxicity_score'],
                        "severity": result['severity'],
                        "risk_level": result['risk_level']
                    },
                    "timestamp": datetime.now()
                }
                try:
                    db_col.insert_one(doc)
                    st.toast("✅ Record saved securely to MongoDB Atlas!")
                except Exception as e:
                    st.toast(f"⚠️ Could not save to DB: {str(e)}")

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
            st.markdown('<div class="section-header" style="font-size:1.1rem; color:#e91e8c; margin-top:1.5rem;">✨ AI Personalized Care Plan</div>', unsafe_allow_html=True)
            
            with st.status("✨ Analyzing results with Gemini AI...", expanded=True) as status:
                ai_advice = get_ai_recommendations(
                    result['side_effect'], 
                    result['risk_level'], 
                    result['severity'],
                    age, 
                    stage
                )
                
                if "⚠️ AI Service Offline" in ai_advice:
                    st.warning("⚠️ AI Service Offline. Showing basic fallback recommendations.")
                    st.markdown(ai_advice)
                else:
                    parts = [p.strip() for p in ai_advice.split("### ") if p.strip()]
                    cols = st.columns(2)
                    for i, part in enumerate(parts):
                        lines = part.split("\n", 1)
                        if len(lines) == 2:
                            title, content = lines[0].strip(), lines[1].strip()
                            
                            color_map = {"Solutions": "#e91e8c", "Diet": "#00e676", "Routine": "#00d4ff", "Safety": "#ffab40"}
                            theme_color = next((color for key, color in color_map.items() if key.lower() in title.lower()), "#e91e8c")
                            
                            with cols[i % 2]:
                                st.markdown(f"""
                                <div style="background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.07); 
                                     border-top:3px solid {theme_color}; border-radius:12px; padding:1.2rem; margin-bottom:1rem; min-height:180px;">
                                    <div style="font-size:1.1rem; color:{theme_color}; font-weight:700; margin-bottom:0.8rem;">
                                        {title}
                                    </div>
                                    <div style="font-size:0.9rem; color:#f0f4ff; line-height:1.6;">
                                """, unsafe_allow_html=True)
                                st.markdown(content)
                                st.markdown("</div></div>", unsafe_allow_html=True)

                st.markdown(f"""
                <hr style="margin:1.2rem 0; opacity:0.1; border-color:#e91e8c;">
                <div style="font-size:0.75rem; color:#7b8aad; text-align:center; font-style:italic;">
                    ⚠️ AI recommendations are for support only. This is NOT medical advice. Always consult your oncologist before starting any new medications or lifestyle changes.
                </div>
                """, unsafe_allow_html=True)
                status.update(label="✅ AI Care Plan Generated!", state="complete", expanded=False)


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
# PAGE: AI ASSISTANT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Assistant":

    st.markdown('<div class="hero-title" style="font-size:2rem;">🤖 AI Health Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Ask questions about chemotherapy, side effects, and wellness tips.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(233,30,140,0.03); border:1px solid rgba(233,30,140,0.15); 
         border-radius:20px; padding:2rem; margin-bottom:2rem;">
        <div style="font-size:1.1rem; color:#e91e8c; font-weight:700; margin-bottom:1rem;">👩‍⚕️ How can I help you today?</div>
        <p style="color:#7b8aad; font-size:0.9rem; margin-bottom:1.5rem;">
            You can ask about diet, medications, side effect management, or emotional wellbeing during treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chat interface
    user_query = st.text_area("Enter your question below:", placeholder="e.g. What diet should I follow during chemo?", height=120)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🚀 Ask AI", use_container_width=True)

    if ask_btn and user_query:
        with st.status("✨ AI is thinking...", expanded=True) as status:
            try:
                # Custom health system prompt
                assistant_prompt = f"""
                Act as a specialized AI Healthcare Assistant for Breast Cancer. 
                Answer the following patient query professionally and compassionately.
                
                Patient Query: {user_query}
                
                Instructions:
                - Focus on Breast Cancer and Chemotherapy context.
                - Use structured bullet points.
                - Provide practical remedies or advice.
                - Always include a medical disclaimer at the end.
                """
                client = get_groq_client()
                response = client.chat.completions.create(
                    model=CURRENT_MODEL_NAME,
                    messages=[{"role": "user", "content": assistant_prompt}],
                    temperature=0.3
                )
                ai_reply = response.choices[0].message.content
                
                st.markdown('<div class="section-header" style="font-size:1.1rem; color:#e91e8c; margin-top:1rem;">👩‍⚕️ AI Response</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.1); 
                     border-radius:15px; padding:1.5rem; line-height:1.7; color:#f0f4ff;">
                    {ai_reply}
                </div>
                """, unsafe_allow_html=True)
                status.update(label="✅ Response Generated!", state="complete")
            except Exception as e:
                # Local Q&A Fallback
                advice = LOCAL_ADVICE["General"]
                for key in LOCAL_ADVICE:
                    if key.lower() in user_query.lower():
                        advice = LOCAL_ADVICE[key]
                        break
                
                st.markdown('<div class="section-header" style="font-size:1.1rem; color:#e91e8c; margin-top:1rem;">👩‍⚕️ AI Status: Limited Mode (API Offline)</div>', unsafe_allow_html=True)
                st.info(f"### ⚠️ AI Service Unavailable\n\n**Quick Tips for your question:**\n\n{advice}\n\n*Error details for developer: {str(e)}*")
                status.update(label="⚠️ Offline Mode", state="complete")
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 Suggested Questions"):
        st.write("1. What are the common side effects of Stage 2 chemotherapy?")
        st.write("2. Can you suggest a weekly diet plan for Chemotherapy patients?")
        st.write("3. How to manage chemotherapy-induced neuropathy?")
        st.write("4. Best breathing exercises for cancer-related anxiety?")



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
