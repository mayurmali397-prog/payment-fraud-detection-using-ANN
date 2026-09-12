import sys
import sys
import streamlit as st

st.warning(f"Python engine running this app: {sys.executable}")

try:
    import tensorflow as tf
    st.success(f"TensorFlow loaded successfully! Version: {tf.__version__}")
except Exception as e:
    st.error(f"Failed to load TensorFlow. Error: {e}")

# ... Keep the rest of your original code below this ...
# ... rest of your imports and code below ...
import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from streamlit_lottie import st_lottie

# ---------------------------------------------------------
# Page Configuration & Custom CSS (React-like UI Styling)
# ---------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Gradient Background & Custom Font */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Modern Glassmorphism Cards */
    div[data-testid="stForm"], div[data-testid="stMetricValue"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Smooth Transition Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(168, 85, 247, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to load Lottie Animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


# Lottie Assets
lottie_shield = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_yzi1544n.json"
)

# ---------------------------------------------------------
# Load Model & Scaler (Cached)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "fraud_detection_model.keras")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_artifacts()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
col_title, col_anim = st.columns([2, 1])

with col_title:
    st.title("🛡️ FraudShield AI")
    st.caption("Real-Time Neural Network Payment Fraud Detection Engine")

with col_anim:
    if lottie_shield:
        st_lottie(lottie_shield, height=140, key="shield")

st.divider()

# Input Form
with st.form("prediction_form"):
    st.subheader("Transaction Metadata")

    c1, c2, c3 = st.columns(3)
    with c1:
        step = st.number_input("Step (Hour of Simulation)", min_value=1, value=1)
        amount = st.number_input(
            "Transaction Amount ($)", min_value=0.0, value=1000.0, step=100.0
        )
        tx_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        )

    with c2:
        oldbalanceOrg = st.number_input(
            "Originator Old Balance", min_value=0.0, value=5000.0
        )
        newbalanceOrig = st.number_input(
            "Originator New Balance", min_value=0.0, value=4000.0
        )

    with c3:
        oldbalanceDest = st.number_input(
            "Destination Old Balance", min_value=0.0, value=0.0
        )
        newbalanceDest = st.number_input(
            "Destination New Balance", min_value=0.0, value=1000.0
        )

    submit_btn = st.form_submit_button("Analyze Transaction")

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
if submit_btn:
    # Map One-Hot Encoded Categoricals exactly as trained in model
    # X columns: step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_CASH_OUT, type_DEBIT, type_TRANSFER
    type_CASH_OUT = 1.0 if tx_type == "CASH_OUT" else 0.0
    type_DEBIT = 1.0 if tx_type == "DEBIT" else 0.0
    type_TRANSFER = 1.0 if tx_type == "TRANSFER" else 0.0

    raw_features = np.array(
        [
            [
                step,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest,
                type_CASH_OUT,
                type_DEBIT,
                type_TRANSFER,
            ]
        ]
    )

    # Scale features using saved scaler
    scaled_features = scaler.transform(raw_features)

    # Predict Probability
    prob = float(model.predict(scaled_features)[0][0])

    # Display Results
    st.subheader("Analysis Results")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.metric(label="Fraud Risk Score", value=f"{prob * 100:.2f}%")

        if prob > 0.5:
            st.error("🚨 WARNING: High probability of Fraud detected!")
        else:
            st.success("✅ LEGITIMATE: Transaction appears safe.")

    with res_col2:
        st.progress(prob)