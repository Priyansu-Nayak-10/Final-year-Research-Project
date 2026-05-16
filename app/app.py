import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")


def load_artifact(name):
    return joblib.load(MODEL_DIR / name)


d_artifact = load_artifact("diabetes_model.pkl")
h_artifact = load_artifact("heart_model.pkl")

DIABETES_DEFAULTS = {
    "age": 53.0,
    "gender": "Female",
    "bmi": 32.75,
    "blood_pressure": 144.0,
    "fasting_glucose_level": 96.0,
    "insulin_level": 13.8,
    "HbA1c_level": 5.5,
    "cholesterol_level": 217.0,
    "triglycerides_level": 173.0,
    "physical_activity_level": "Low",
    "daily_calorie_intake": 2385.0,
    "sugar_intake_grams_per_day": 58.3,
    "sleep_hours": 7.1,
    "stress_level": 5.0,
    "family_history_diabetes": "No",
    "waist_circumference_cm": 104.6,
}

HEART_DEFAULTS = {
    "age": 54.0,
    "bmi": 28.4,
    "systolic_bp": 147.0,
    "diastolic_bp": 96.0,
    "cholesterol_mg_dl": 240.0,
    "resting_heart_rate": 74.0,
    "smoking_status": "Never",
    "daily_steps": 5460.0,
    "stress_level": 5.0,
    "physical_activity_hours_per_week": 2.6,
    "sleep_hours": 6.9,
    "family_history_heart_disease": "No",
    "diet_quality_score": 5.0,
    "alcohol_units_per_week": 2.8,
}


# ---------------- CORE ----------------
def is_missing(value):
    return value is None or value == "Select"


def prepare_inputs(raw_inputs, required_fields, defaults):
    missing = [label for feature, label in required_fields if is_missing(raw_inputs.get(feature))]
    if missing:
        st.error("Required fields missing: " + ", ".join(missing))
        st.stop()

    filled_inputs = {}
    autofilled = []
    for feature, default_value in defaults.items():
        val = raw_inputs.get(feature)
        if is_missing(val):
            filled_inputs[feature] = default_value
            autofilled.append(feature)
        else:
            filled_inputs[feature] = val

    return pd.DataFrame([filled_inputs]), autofilled


def run_prediction(data, artifact):
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    selector = artifact["selector"]
    input_cols = artifact["numeric_cols"] + artifact["categorical_cols"]

    try:
        transformed = preprocessor.transform(data[input_cols])
        selected = selector.transform(transformed)
        pred = model.predict(selected)[0]
        prob = model.predict_proba(selected)[0][1]
        return pred, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()


# ---------------- RESULT ----------------
def show_result(pred, prob):
    color = "red" if pred else "green"
    st.markdown(
        f"""
    <div style="padding:18px;border-radius:10px;background:#111;">
        <h3 style="color:{color};">Risk: {"High" if pred else "Low"}</h3>
        <p style="color:#fff;">Probability: {prob:.2f}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def interpret(prob):
    if prob > 0.8:
        st.warning("Very high risk")
    elif prob > 0.5:
        st.info("Moderate risk")
    else:
        st.success("Low risk")


# ---------------- UI ----------------
st.set_page_config(page_title="AI Health Predictor", layout="wide")

# REMOVE +/- BUTTONS (clean fix)
st.markdown(
    """
<style>
div[data-testid="stNumberInput"] button {display:none;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("TrustMed - AI Disease Predictor")

option = st.sidebar.selectbox("Select", ["Diabetes", "Heart Disease"])

# ================= DIABETES =================
if option == "Diabetes":
    st.subheader("Diabetes Prediction")
    st.caption(
        "Only key fields are required. Skipped optional fields are auto-filled from training medians/modes."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 1, 100, value=None, placeholder="Enter age")
        gender = st.selectbox("Gender (Optional)", ["Select", "Female", "Male"])
        bmi = st.number_input("BMI", 10.0, 60.0, value=None, placeholder="e.g. 22.5")
        bp = st.number_input("Blood Pressure", 60.0, 240.0, value=None, placeholder="e.g. 120")
        glucose = st.number_input("Fasting Glucose", 50.0, 350.0, value=None, placeholder="e.g. 100")

    with c2:
        insulin = st.number_input("Insulin (Optional)", 0.0, 400.0, value=None, placeholder="e.g. 80")
        hba1c = st.number_input("HbA1c", 3.0, 20.0, value=None, placeholder="e.g. 5.5")
        chol = st.number_input("Cholesterol (Optional)", 80.0, 500.0, value=None, placeholder="e.g. 180")
        trig = st.number_input("Triglycerides (Optional)", 50.0, 700.0, value=None, placeholder="e.g. 150")
        activity = st.selectbox("Activity (Optional)", ["Select", "Low", "Moderate", "High"])

    with c3:
        calories = st.number_input("Calories (Optional)", 500.0, 6000.0, value=None, placeholder="e.g. 2000")
        sugar = st.number_input("Sugar (Optional)", 0.0, 500.0, value=None, placeholder="e.g. 60")
        sleep = st.number_input("Sleep (Optional)", 0.0, 14.0, value=None, placeholder="6-8")
        stress = st.number_input("Stress (Optional)", 0.0, 10.0, value=None, placeholder="e.g. 5")
        family = st.selectbox("Family History (Optional)", ["Select", "No", "Yes"])
        waist = st.number_input(
            "Waist Circumference (Optional, cm)", 40.0, 200.0, value=None, placeholder="e.g. 90"
        )

    if st.button("Predict Diabetes", use_container_width=True):
        raw_data = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "blood_pressure": bp,
            "fasting_glucose_level": glucose,
            "insulin_level": insulin,
            "HbA1c_level": hba1c,
            "cholesterol_level": chol,
            "triglycerides_level": trig,
            "physical_activity_level": activity,
            "daily_calorie_intake": calories,
            "sugar_intake_grams_per_day": sugar,
            "sleep_hours": sleep,
            "stress_level": stress,
            "family_history_diabetes": family,
            "waist_circumference_cm": waist,
        }

        required_fields = [
            ("age", "Age"),
            ("bmi", "BMI"),
            ("blood_pressure", "Blood Pressure"),
            ("fasting_glucose_level", "Fasting Glucose"),
            ("HbA1c_level", "HbA1c"),
        ]

        data, autofilled = prepare_inputs(raw_data, required_fields, DIABETES_DEFAULTS)
        if autofilled:
            st.info("Optional auto-filled fields: " + ", ".join(autofilled))

        pred, prob = run_prediction(data, d_artifact)
        show_result(pred, prob)
        interpret(prob)

# ================= HEART =================
else:
    st.subheader("Heart Disease Prediction")
    st.caption(
        "Only key fields are required. Skipped optional fields are auto-filled from training medians/modes."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 1, 100, value=None, placeholder="Enter age")
        bmi = st.number_input("BMI", 10.0, 60.0, value=None, placeholder="e.g. 24")
        sys = st.number_input("Systolic BP", 70.0, 260.0, value=None, placeholder="e.g. 120")
        dia = st.number_input("Diastolic BP", 40.0, 160.0, value=None, placeholder="e.g. 80")
        chol = st.number_input("Cholesterol", 80.0, 500.0, value=None, placeholder="e.g. 190")

    with c2:
        hr = st.number_input("Heart Rate", 35.0, 220.0, value=None, placeholder="e.g. 72")
        smoking = st.selectbox("Smoking (Optional)", ["Select", "Never", "Former", "Current"])
        steps = st.number_input("Steps (Optional)", 0, 60000, value=None, placeholder="e.g. 7000")
        stress = st.number_input("Stress (Optional)", 0.0, 10.0, value=None, placeholder="e.g. 5")
        activity = st.number_input("Activity hrs (Optional)", 0.0, 60.0, value=None, placeholder="e.g. 3")

    with c3:
        sleep = st.number_input("Sleep (Optional)", 0.0, 14.0, value=None, placeholder="6-8")
        family = st.selectbox("Family History (Optional)", ["Select", "No", "Yes"])
        diet = st.number_input("Diet Score (Optional)", 0.0, 10.0, value=None, placeholder="e.g. 6")
        alcohol = st.number_input("Alcohol (Optional)", 0.0, 80.0, value=None, placeholder="e.g. 2")
        dummy = st.empty()  # keeps grid balanced

    if st.button("Predict Heart Risk", use_container_width=True):
        raw_data = {
            "age": age,
            "bmi": bmi,
            "systolic_bp": sys,
            "diastolic_bp": dia,
            "cholesterol_mg_dl": chol,
            "resting_heart_rate": hr,
            "smoking_status": smoking,
            "daily_steps": steps,
            "stress_level": stress,
            "physical_activity_hours_per_week": activity,
            "sleep_hours": sleep,
            "family_history_heart_disease": family,
            "diet_quality_score": diet,
            "alcohol_units_per_week": alcohol,
        }

        required_fields = [
            ("age", "Age"),
            ("bmi", "BMI"),
            ("systolic_bp", "Systolic BP"),
            ("diastolic_bp", "Diastolic BP"),
            ("cholesterol_mg_dl", "Cholesterol"),
            ("resting_heart_rate", "Heart Rate"),
        ]

        data, autofilled = prepare_inputs(raw_data, required_fields, HEART_DEFAULTS)
        if autofilled:
            st.info("Optional auto-filled fields: " + ", ".join(autofilled))

        pred, prob = run_prediction(data, h_artifact)
        show_result(pred, prob)
        interpret(prob)
