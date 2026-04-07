import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ---------------- PATH SETUP ----------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return {
        "diabetes": joblib.load(MODELS_DIR / "diabetes_model.pkl"),
        "heart": joblib.load(MODELS_DIR / "heart_model.pkl"),
        "stroke": joblib.load(MODELS_DIR / "stroke_model.pkl"),
    }

artifacts = load_models()

def unpack(model_data):
    return (
        model_data["model"],
        model_data.get("scaler"),
        model_data.get("selector"),
        model_data.get("threshold", 0.5),
        model_data.get("imputer"),
        model_data.get("feature_columns"),
    )

# Unpack
d_model, d_scaler, d_selector, d_thresh, _, _ = unpack(artifacts["diabetes"])
h_model, h_scaler, h_selector, h_thresh, _, h_features = unpack(artifacts["heart"])
s_model, s_scaler, s_selector, s_thresh, s_imputer, s_features = unpack(artifacts["stroke"])

h_mappings = artifacts["heart"].get("categorical_mappings", {})
h_features = h_features or [
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

# ---------------- HELPERS ----------------
def process_pipeline(x, scaler, selector, imputer=None):
    if imputer:
        x = imputer.transform(x)
    if scaler:
        x = scaler.transform(x)
    if selector:
        x = selector.transform(x)
    return x

def risk_label(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"

# ---------------- UI ----------------
st.set_page_config(page_title="AI Health Predictor", layout="wide")

st.sidebar.title("Disease Predictor")
disease = st.sidebar.radio("Select Disease", ["Diabetes", "Heart Disease", "Stroke"])

st.title("AI-Based Disease Risk Prediction System")

# ---------------- DIABETES ----------------
if disease == "Diabetes":
    st.subheader("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        bmi = st.number_input("BMI", 10.0, 60.0)

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1])

    hba1c = st.number_input("HbA1c Level")
    glucose = st.number_input("Blood Glucose Level")

    if st.button("Predict Diabetes"):
        # ONLY selected features
        x = np.array([[age, hypertension, bmi, hba1c, glucose]])
        x = process_pipeline(x, d_scaler, d_selector)

        prob = d_model.predict_proba(x)[0][1]
        pred = int(prob >= d_thresh)

        st.subheader("Result")
        st.progress(int(prob * 100))
        st.markdown(f"### Risk: **{risk_label(prob)}**")
        st.markdown(f"### Probability: {round(prob * 100, 2)}%")
        st.success("Positive" if pred else "Negative")


# ---------------- HEART ----------------
elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        bmi = st.number_input("BMI", 10.0, 60.0)
        systolic_bp = st.number_input("Systolic BP", 80, 260)
        diastolic_bp = st.number_input("Diastolic BP", 50, 180)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 500)

    with col2:
        heart_rate = st.number_input("Resting Heart Rate", 40, 180)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        steps = st.number_input("Daily Steps", 0, 40000)
        stress_level = st.slider("Stress Level", 1, 10, 5)
        activity = st.number_input("Physical Activity (hrs/week)", 0.0, 40.0)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 14.0, 7.0)
        family_history = st.selectbox("Family History", ["No", "Yes"])
        diet = st.slider("Diet Quality Score", 1, 10, 5)
        alcohol_units = st.number_input("Alcohol Units/Week", 0.0, 40.0, 0.0)

    if st.button("Predict Heart Disease"):
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

        # Apply mappings
        for col, mapping in h_mappings.items():
            if col in h_dict:
                h_dict[col] = mapping.get(h_dict[col], 0)

        x = np.array([[h_dict.get(col, 0) for col in h_features]], dtype=float)
        x = process_pipeline(x, h_scaler, h_selector)

        prob = h_model.predict_proba(x)[0][1]
        pred = int(prob >= h_thresh)

        st.subheader("Result")
        st.progress(int(prob * 100))
        st.markdown(f"### Risk: **{risk_label(prob)}**")
        st.markdown(f"### Probability: {round(prob * 100, 2)}%")
        st.success("Positive" if pred else "Negative")


# ---------------- STROKE ----------------
elif disease == "Stroke":
    st.subheader("Stroke Prediction")

    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox("Sex", [0, 1])
        age = st.number_input("Age", 1, 120)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])

    with col2:
        married = st.selectbox("Married", [0, 1])
        work = st.selectbox("Work Type", [0, 1, 2, 3, 4])
        glucose = st.number_input("Avg Glucose Level")
        smoking = st.selectbox("Smoking", [0, 1, 2, 3])

    if st.button("Predict Stroke"):
        s_dict = {
            "sex": sex,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": married,
            "work_type": work,
            "avg_glucose_level": glucose,
            "smoking_status": smoking,
        }

        x = np.array([[s_dict[col] for col in s_features]])
        x = process_pipeline(x, s_scaler, s_selector, s_imputer)

        prob = s_model.predict_proba(x)[0][1]
        pred = int(prob >= s_thresh)

        st.subheader("Result")
        st.progress(int(prob * 100))
        st.markdown(f"### Risk: **{risk_label(prob)}**")
        st.markdown(f"### Probability: {round(prob * 100, 2)}%")
        st.success("Positive" if pred else "Negative")
