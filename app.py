from __future__ import annotations

import streamlit as st

from src.predict.predict_diabetes import predict_diabetes

SMOKING_OPTIONS = {
    "Never": "never",
    "Former": "former",
    "Current": "current",
    "Ever": "ever",
    "Not Current": "not current",
    "No Info": "no info",
}


st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("Diabetes Risk Predictor")
st.caption("Model output is a screening estimate, not a medical diagnosis.")

st.write("Enter details below to calculate diabetes risk probability.")

with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.5, step=0.1)
        hba1c_level = st.number_input(
            "HbA1c Level", min_value=3.0, max_value=15.0, value=5.8, step=0.1
        )

    with col2:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        smoking_label = st.selectbox("Smoking History", list(SMOKING_OPTIONS.keys()), index=0)
        blood_glucose_level = st.number_input(
            "Blood Glucose Level", min_value=40, max_value=500, value=120, step=1
        )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": SMOKING_OPTIONS[smoking_label],
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level,
    }

    try:
        result = predict_diabetes(input_data)
        probability_pct = result["probability"] * 100

        st.subheader("Prediction Result")
        st.metric("Predicted Probability", f"{probability_pct:.2f}%")
        st.caption(f"Decision threshold: {result['threshold']:.2f}")

        if result["prediction"] == 1:
            st.error(f"{result['label']} (Positive class)")
        else:
            st.success(f"{result['label']} (Negative class)")

        with st.expander("Model Details"):
            st.write(f"Model: `{result['model_name']}`")
            st.write(f"Scaler applied: `{result['uses_scaler']}`")

        with st.expander("Model-ready Input (13 features)"):
            st.json(result["model_input"])
    except Exception as exc:
        st.exception(exc)
