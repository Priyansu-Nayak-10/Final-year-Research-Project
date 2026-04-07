import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ---------------- PATH SETUP ----------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

DIABETES_FALLBACK_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "hba1c_level",
    "blood_glucose_level",
]

HEART_FALLBACK_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_mg_dl",
    "resting_heart_rate",
    "smoking_status",
    "daily_steps",
    "stress_level",
    "physical_activity_hours_per_week",
    "sleep_hours",
    "family_history",
    "diet_quality_score",
    "alcohol_units_per_week",
]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return {
        "diabetes": joblib.load(MODELS_DIR / "diabetes_model.pkl"),
        "heart": joblib.load(MODELS_DIR / "heart_model.pkl"),
        "stroke": joblib.load(MODELS_DIR / "stroke_model.pkl"),
    }


def unpack(model_data):
    return (
        model_data["model"],
        model_data.get("scaler"),
        model_data.get("selector"),
        model_data.get("threshold", 0.5),
        model_data.get("imputer"),
        model_data.get("feature_columns"),
    )


def resolve_feature_order(saved_features, fallback_features):
    return saved_features if saved_features else fallback_features


def process_pipeline(x, scaler, selector, imputer=None):
    if imputer is not None:
        x = imputer.transform(x)
    if scaler is not None:
        x = scaler.transform(x)
    if selector is not None:
        x = selector.transform(x)
    return x


def ordered_array_from_dict(feature_dict, feature_order):
    missing = [feature for feature in feature_order if feature not in feature_dict]
    if missing:
        raise KeyError(f"Missing required features: {missing}")
    return np.array([[feature_dict[feature] for feature in feature_order]], dtype=float)


def risk_label(prob):
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


def render_result(prob, pred):
    st.markdown("### Prediction Result")
    st.progress(int(prob * 100))
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Tier", risk_label(prob))
    col_b.metric("Probability", f"{prob * 100:.2f}%")
    col_c.metric("Decision", "Positive" if pred else "Negative")


artifacts = load_models()

# ---------------- UNPACK ----------------
d_model, d_scaler, d_selector, d_thresh, _, d_features_saved = unpack(artifacts["diabetes"])
h_model, h_scaler, h_selector, h_thresh, _, h_features_saved = unpack(artifacts["heart"])
s_model, s_scaler, s_selector, s_thresh, s_imputer, s_features_saved = unpack(artifacts["stroke"])

d_features = resolve_feature_order(d_features_saved, DIABETES_FALLBACK_FEATURES)
h_features = resolve_feature_order(h_features_saved, HEART_FALLBACK_FEATURES)
s_features = s_features_saved or []
h_mappings = artifacts["heart"].get("categorical_mappings", {})

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Health Predictor", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --accent: #0f766e;
        --accent-dark: #115e59;
        --ink: #16333f;
        --panel: #ffffff;
        --line: #cfe1dc;
        --soft: #edf6f4;
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Trebuchet MS", "Arial", sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 650px at 105% -8%, #d6eef5 0%, rgba(214, 238, 245, 0) 45%),
            radial-gradient(900px 500px at -10% 0%, #eaf7ef 0%, rgba(234, 247, 239, 0) 40%),
            linear-gradient(180deg, #f4faf8 0%, #eff6f7 46%, #f8fbff 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8f2f4 0%, #eff7f6 100%);
        border-right: 1px solid #d0dfe4;
    }
    [data-testid="stSidebar"] * {
        color: #1c4350;
    }
    .hero {
        background: linear-gradient(120deg, #0f3c4f 0%, #1d6273 55%, #2f8c7a 100%);
        border-radius: 16px;
        padding: 22px 24px 20px 24px;
        color: #ffffff;
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 14px 30px rgba(16, 61, 84, 0.18);
    }
    .hero h2 {
        margin: 0;
        font-size: 1.58rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 9px 0 0 0;
        font-size: 0.98rem;
        opacity: 0.94;
    }
    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 18px 8px 18px;
        box-shadow: 0 7px 20px rgba(17, 92, 103, 0.08);
        margin-bottom: 16px;
    }
    [data-testid="metric-container"] {
        background: var(--soft);
        border: 1px solid #d8e8e2;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(12, 58, 72, 0.06);
    }
    div[data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, var(--accent), #13857d);
        color: #ffffff;
        border: 0;
        border-radius: 10px;
        padding: 0.56rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, var(--accent-dark), #0f766e);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2>AI-Based Disease Risk Prediction System</h2>
      <p>Clinical risk screening dashboard for diabetes, cardiovascular risk, and stroke.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Clinical Models")
disease = st.sidebar.radio("Choose model", ["Diabetes", "Heart Disease", "Stroke"])
st.sidebar.caption("Prediction uses saved model artifacts and thresholds.")

# ---------------- DIABETES ----------------
if disease == "Diabetes":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Diabetes Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 40)
        hypertension = st.selectbox("Hypertension", [0, 1], help="0 = No, 1 = Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="0 = No, 1 = Yes")
    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        hba1c = st.number_input("HbA1c Level", 1.0, 20.0, 5.8)
        glucose = st.number_input("Blood Glucose Level", 40.0, 400.0, 110.0)

    if st.button("Predict Diabetes", type="primary"):
        try:
            d_dict = {
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "bmi": bmi,
                "hba1c_level": hba1c,
                "blood_glucose_level": glucose,
            }
            x = ordered_array_from_dict(d_dict, d_features)
            x = process_pipeline(x, d_scaler, d_selector)
            prob = d_model.predict_proba(x)[0][1]
            pred = int(prob >= d_thresh)
            render_result(prob, pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HEART ----------------
elif disease == "Heart Disease":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Heart Disease Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
        systolic_bp = st.number_input("Systolic BP", 80, 260, 124)
        diastolic_bp = st.number_input("Diastolic BP", 50, 180, 82)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 500, 200)
        heart_rate = st.number_input("Resting Heart Rate", 40, 180, 72)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    with col2:
        steps = st.number_input("Daily Steps", 0, 40000, 7000)
        stress_level = st.slider("Stress Level", 1, 10, 5)
        activity = st.number_input("Physical Activity (hrs/week)", 0.0, 40.0, 3.0)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 14.0, 7.0)
        family_history = st.selectbox("Family History", ["No", "Yes"])
        diet = st.slider("Diet Quality Score", 1, 10, 5)
        alcohol_units = st.number_input("Alcohol Units/Week", 0.0, 40.0, 1.0)

    if st.button("Predict Heart Disease", type="primary"):
        try:
            h_dict = {
                "age": age,
                "bmi": bmi,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "cholesterol_mg_dl": cholesterol,
                "resting_heart_rate": heart_rate,
                "smoking_status": smoking,
                "daily_steps": steps,
                "stress_level": stress_level,
                "physical_activity_hours_per_week": activity,
                "sleep_hours": sleep_hours,
                "family_history": family_history,
                "diet_quality_score": diet,
                "alcohol_units_per_week": alcohol_units,
            }
            for col, mapping in h_mappings.items():
                if col in h_dict:
                    h_dict[col] = mapping.get(h_dict[col], 0)
            x = ordered_array_from_dict(h_dict, h_features)
            x = process_pipeline(x, h_scaler, h_selector)
            prob = h_model.predict_proba(x)[0][1]
            pred = int(prob >= h_thresh)
            render_result(prob, pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- STROKE ----------------
elif disease == "Stroke":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Stroke Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", [0, 1], help="Encoded numeric value")
        age = st.number_input("Age", 1, 120, 50)
        hypertension = st.selectbox("Hypertension", [0, 1], help="0 = No, 1 = Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="0 = No, 1 = Yes")
        married = st.selectbox("Ever Married", [0, 1], help="0 = No, 1 = Yes")
    with col2:
        work = st.selectbox("Work Type", [0, 1, 2, 3, 4])
        residence_type = st.selectbox("Residence Type", [0, 1], help="Encoded numeric value")
        glucose = st.number_input("Average Glucose Level", 40.0, 400.0, 105.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 26.0)
        smoking = st.selectbox("Smoking Status", [0, 1], help="Using dataset-consistent values")

    if st.button("Predict Stroke", type="primary"):
        try:
            s_dict = {
                "sex": sex,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": married,
                "work_type": work,
                "residence_type": residence_type,
                "avg_glucose_level": glucose,
                "bmi": bmi,
                "smoking_status": smoking,
            }
            x = ordered_array_from_dict(s_dict, s_features)
            x = process_pipeline(x, s_scaler, s_selector, s_imputer)
            prob = s_model.predict_proba(x)[0][1]
            pred = int(prob >= s_thresh)
            render_result(prob, pred)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)
