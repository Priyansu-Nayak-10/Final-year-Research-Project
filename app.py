import streamlit as st
import numpy as np
import joblib

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
heart_model = joblib.load("models/heart_model.pkl")

diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")

st.title("AI Disease Risk Predictor")

# ======================
# USER INPUT
# ======================

age = st.slider("Age", 1, 100, 25)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
bp = st.number_input("Blood Pressure", 80, 200, 120)
glucose = st.number_input("Glucose Level", 50, 300, 100)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)

smoking = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

# ======================
# PREDICT
# ======================

if st.button("Predict"):

    # Diabetes input
    d_input = np.array([[age, bmi, bp, glucose]])
    d_input = diabetes_scaler.transform(d_input)
    d_prob = diabetes_model.predict_proba(d_input)[0][1]

    # Heart input
    h_input = np.array([[age, bp, cholesterol, smoking]])
    h_input = heart_scaler.transform(h_input)
    h_prob = heart_model.predict_proba(h_input)[0][1]

    st.subheader("Results")

    st.write(f"Diabetes Risk: {round(d_prob*100,2)}%")
    st.write(f"Heart Disease Risk: {round(h_prob*100,2)}%")

    # Simple interpretation
    if d_prob > 0.7:
        st.error("High Diabetes Risk")
    elif d_prob > 0.4:
        st.warning("Moderate Diabetes Risk")
    else:
        st.success("Low Diabetes Risk")

    if h_prob > 0.7:
        st.error("High Heart Risk")
    elif h_prob > 0.4:
        st.warning("Moderate Heart Risk")
    else:
        st.success("Low Heart Risk")