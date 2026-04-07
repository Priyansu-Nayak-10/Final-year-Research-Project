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
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: "Manrope", "Trebuchet MS", "Segoe UI", sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top right, #dff5e8 0%, #f6fbf8 42%, #ffffff 100%);
    }
    .hero {
        background: linear-gradient(120deg, #123c4a 0%, #1e5f74 55%, #2b7a78 100%);
        border-radius: 16px;
        padding: 22px 24px;
        color: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 12px 28px rgba(18, 60, 74, 0.20);
    }
    .hero h2 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 8px 0 0 0;
        font-size: 0.98rem;
        opacity: 0.95;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #d8e6df;
        border-radius: 14px;
        padding: 10px 16px 4px 16px;
        box-shadow: 0 6px 18px rgba(30, 95, 116, 0.08);
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2>AI-Based Disease Risk Prediction System</h2>
      <p>Production-style multi-model screening for diabetes, cardiovascular risk, and stroke.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Disease Selector")
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
