import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

def load_artifact(name):
    art = joblib.load(MODEL_DIR / name)
    return art["model"], art["scaler"], art["encoders"], art["features"]

d_model, d_scaler, d_encoders, d_features = load_artifact("diabetes_model.pkl")
h_model, h_scaler, h_encoders, h_features = load_artifact("heart_model.pkl")

# ---------------- CORE ----------------
def run_prediction(data, model, scaler, encoders, features):
    for col, enc in encoders.items():
        if col in data:
            try:
                data[col] = enc.transform(data[col])
            except:
                st.error(f"Invalid value for {col}")
                st.stop()
    try:
        data = scaler.transform(data[features])
        return model.predict(data)[0], model.predict_proba(data)[0][1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

def validate_required(fields):
    for k, v in fields.items():
        if v is None or v == "Select":
            st.error(f"{k} is required")
            st.stop()

# ---------------- RESULT ----------------
def show_result(pred, prob):
    color = "red" if pred else "green"
    st.markdown(f"""
    <div style="padding:18px;border-radius:10px;background:#111;">
        <h3 style="color:{color};">Risk: {"High" if pred else "Low"}</h3>
        <p style="color:#fff;">Probability: {prob:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

def interpret(prob):
    if prob > 0.8: st.warning("Very high risk")
    elif prob > 0.5: st.info("Moderate risk")
    else: st.success("Low risk")

# ---------------- UI ----------------
st.set_page_config(page_title="AI Health Predictor", layout="wide")

# REMOVE +/- BUTTONS (clean fix)
st.markdown("""
<style>
div[data-testid="stNumberInput"] button {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("TrustMed - AI Disease Predictor")

option = st.sidebar.selectbox("Select", ["Diabetes", "Heart Disease"])

# ================= DIABETES =================
if option == "Diabetes":
    st.subheader("Diabetes Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 1, 100, value=None, placeholder="Enter age")
        gender = st.selectbox("Gender", ["Select"] + list(d_encoders["gender"].classes_))
        bmi = st.number_input("BMI", 10.0, 60.0, value=None, placeholder="e.g. 22.5")
        bp = st.number_input("Blood Pressure", 60.0, 240.0, value=None, placeholder="e.g. 120")
        glucose = st.number_input("Fasting Glucose", 50.0, 350.0, value=None, placeholder="e.g. 100")

    with c2:
        insulin = st.number_input("Insulin", 0.0, 400.0, value=None, placeholder="e.g. 80")
        hba1c = st.number_input("HbA1c", 3.0, 20.0, value=None, placeholder="e.g. 5.5")
        chol = st.number_input("Cholesterol", 80.0, 500.0, value=None, placeholder="e.g. 180")
        trig = st.number_input("Triglycerides", 50.0, 700.0, value=None, placeholder="e.g. 150")
        activity = st.selectbox("Activity", ["Select"] + list(d_encoders["physical_activity_level"].classes_))

    with c3:
        calories = st.number_input("Calories", 500.0, 6000.0, value=None, placeholder="e.g. 2000")
        sugar = st.number_input("Sugar", 0.0, 500.0, value=None, placeholder="e.g. 60")
        sleep = st.number_input("Sleep", 0.0, 14.0, value=None, placeholder="6-8")
        stress = st.number_input("Stress", 0.0, 10.0, value=None, placeholder="e.g. 5")
        family = st.selectbox("Family History", ["Select"] + list(d_encoders["family_history_diabetes"].classes_))
        waist = st.number_input("Waist Circumference (cm)", 40.0, 200.0, value=None, placeholder="e.g. 90")

    if st.button("Predict Diabetes", use_container_width=True):
        validate_required({
            "Age": age, "Gender": gender, "BMI": bmi, "BP": bp,
            "Glucose": glucose, "Insulin": insulin, "HbA1c": hba1c,
            "Chol": chol, "Trig": trig, "Activity": activity,
            "Calories": calories, "Sugar": sugar, "Sleep": sleep,
            "Stress": stress, "Family": family, "Waist": waist
        })

        data = pd.DataFrame([{
            "age": age, "gender": gender, "bmi": bmi,
            "blood_pressure": bp, "fasting_glucose_level": glucose,
            "insulin_level": insulin, "HbA1c_level": hba1c,
            "cholesterol_level": chol, "triglycerides_level": trig,
            "physical_activity_level": activity,
            "daily_calorie_intake": calories,
            "sugar_intake_grams_per_day": sugar,
            "sleep_hours": sleep, "stress_level": stress,
            "family_history_diabetes": family,
            "waist_circumference_cm": waist
        }])

        pred, prob = run_prediction(data, d_model, d_scaler, d_encoders, d_features)
        show_result(pred, prob)
        interpret(prob)

# ================= HEART =================
else:
    st.subheader("Heart Disease Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 1, 100, value=None, placeholder="Enter age")
        bmi = st.number_input("BMI", 10.0, 60.0, value=None, placeholder="e.g. 24")
        sys = st.number_input("Systolic BP", 70.0, 260.0, value=None, placeholder="e.g. 120")
        dia = st.number_input("Diastolic BP", 40.0, 160.0, value=None, placeholder="e.g. 80")
        chol = st.number_input("Cholesterol", 80.0, 500.0, value=None, placeholder="e.g. 190")

    with c2:
        hr = st.number_input("Heart Rate", 35.0, 220.0, value=None, placeholder="e.g. 72")
        smoking = st.selectbox("Smoking", ["Select"] + list(h_encoders["smoking_status"].classes_))
        steps = st.number_input("Steps", 0, 60000, value=None, placeholder="e.g. 7000")
        stress = st.number_input("Stress", 0.0, 10.0, value=None, placeholder="e.g. 5")
        activity = st.number_input("Activity hrs", 0.0, 60.0, value=None, placeholder="e.g. 3")

    with c3:
        sleep = st.number_input("Sleep", 0.0, 14.0, value=None, placeholder="6-8")
        family = st.selectbox("Family History", ["Select"] + list(h_encoders["family_history_heart_disease"].classes_))
        diet = st.number_input("Diet Score", 0.0, 10.0, value=None, placeholder="e.g. 6")
        alcohol = st.number_input("Alcohol", 0.0, 80.0, value=None, placeholder="e.g. 2")
        dummy = st.empty()  # keeps grid balanced

    if st.button("Predict Heart Risk", use_container_width=True):
        validate_required({
            "Age": age, "BMI": bmi, "Sys": sys, "Dia": dia,
            "Chol": chol, "HR": hr, "Smoking": smoking,
            "Steps": steps, "Stress": stress, "Activity": activity,
            "Sleep": sleep, "Family": family, "Diet": diet, "Alcohol": alcohol
        })

        data = pd.DataFrame([{
            "age": age, "bmi": bmi, "systolic_bp": sys,
            "diastolic_bp": dia, "cholesterol_mg_dl": chol,
            "resting_heart_rate": hr, "smoking_status": smoking,
            "daily_steps": steps, "stress_level": stress,
            "physical_activity_hours_per_week": activity,
            "sleep_hours": sleep,
            "family_history_heart_disease": family,
            "diet_quality_score": diet,
            "alcohol_units_per_week": alcohol
        }])

        pred, prob = run_prediction(data, h_model, h_scaler, h_encoders, h_features)
        show_result(pred, prob)
        interpret(prob)
