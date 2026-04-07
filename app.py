import streamlit as st

from src.predict import predict_risk


st.set_page_config(page_title="Early Disease Risk Prediction", layout="wide")

st.markdown(
    """
    <style>
    .title-wrap {padding: 0.2rem 0 0.8rem 0;}
    .subtitle {color: #5b6470; font-size: 1rem; margin-top: -8px;}
    .section-gap {margin-top: 1rem;}
    .risk-card {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid #e6e9ef;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .risk-score {font-size: 2rem; font-weight: 700; margin: 0;}
    .risk-label {font-size: 1rem; font-weight: 600; margin: 0.2rem 0 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-wrap">', unsafe_allow_html=True)
st.title("AI-Based Chronic Disease Risk Prediction System")
st.markdown(
    '<div class="subtitle">Predict risk for Diabetes, Heart Disease, COPD, and CKD</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

disease_options = ["Diabetes", "Heart Disease", "COPD", "Chronic Kidney Disease"]
disease_key_map = {
    "Diabetes": "diabetes",
    "Heart Disease": "heart",
    "COPD": "copd",
    "Chronic Kidney Disease": "ckd",
}

disease_label = st.selectbox("Select Disease", disease_options, index=0)

with st.form("risk_form"):
    st.markdown("### Basic Info")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.5, step=0.1)
        blood_pressure = st.number_input(
            "Blood Pressure", min_value=80, max_value=200, value=120, step=1
        )

    with c2:
        glucose = st.number_input("Glucose", min_value=40, max_value=500, value=110, step=1)
        smoking = st.selectbox("Smoking", ["No", "Yes"], index=0)
        physical_activity = st.selectbox(
            "Physical Activity", ["Low", "Moderate", "High"], index=1
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### Disease-Specific Inputs")
    d1, d2 = st.columns(2)

    disease_inputs = {}

    if disease_label == "Diabetes":
        with d1:
            insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0
            )
        with d2:
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                format="%.2f",
            )

        disease_inputs.update(
            {
                "Insulin": insulin,
                "SkinThickness": skin_thickness,
                "DiabetesPedigreeFunction": dpf,
            }
        )

    elif disease_label == "Heart Disease":
        with d1:
            chest_pain = st.selectbox("ChestPainType", ["TA", "ATA", "NAP", "ASY"], index=3)
            cholesterol = st.number_input(
                "Cholesterol", min_value=50, max_value=700, value=200, step=1
            )
            resting_bp = st.number_input("RestingBP", min_value=80, max_value=220, value=120, step=1)
            max_hr = st.number_input("MaxHR", min_value=50, max_value=220, value=150, step=1)
        with d2:
            exercise_angina = st.selectbox("ExerciseAngina", ["No", "Yes"], index=0)
            resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"], index=0)
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"], index=1)

        disease_inputs.update(
            {
                "ChestPainType": chest_pain,
                "Cholesterol": cholesterol,
                "RestingBP": resting_bp,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "RestingECG": resting_ecg,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope,
            }
        )

    elif disease_label == "COPD":
        with d1:
            smoking_years = st.number_input(
                "Smoking Years", min_value=0, max_value=80, value=10, step=1
            )
            breathlessness = st.slider("Breathlessness Level", min_value=1, max_value=5, value=2)
        with d2:
            chronic_cough = st.selectbox("Chronic Cough", ["No", "Yes"], index=0)
            pollution = st.selectbox(
                "Air Pollution Exposure", ["Low", "Medium", "High"], index=1
            )

        disease_inputs.update(
            {
                "SmokingYears": smoking_years,
                "BreathlessnessLevel": breathlessness,
                "ChronicCough": chronic_cough,
                "AirPollutionExposure": pollution,
            }
        )

    elif disease_label == "Chronic Kidney Disease":
        with d1:
            serum_creatinine = st.number_input(
                "Serum Creatinine", min_value=0.1, max_value=20.0, value=1.0, step=0.1
            )
            blood_urea = st.number_input("Blood Urea", min_value=5.0, max_value=300.0, value=35.0, step=1.0)
            albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            hemoglobin = st.number_input(
                "Hemoglobin", min_value=3.0, max_value=20.0, value=13.0, step=0.1
            )
        with d2:
            sodium = st.number_input("Sodium", min_value=100.0, max_value=170.0, value=138.0, step=1.0)
            potassium = st.number_input("Potassium", min_value=2.0, max_value=8.0, value=4.2, step=0.1)
            appetite = st.selectbox("Appetite", ["Good", "Poor"], index=0)
            pedal_edema = st.selectbox("Pedal Edema", ["No", "Yes"], index=0)

        disease_inputs.update(
            {
                "SerumCreatinine": serum_creatinine,
                "BloodUrea": blood_urea,
                "Albumin": albumin,
                "Hemoglobin": hemoglobin,
                "Sodium": sodium,
                "Potassium": potassium,
                "Appetite": appetite,
                "PedalEdema": pedal_edema,
            }
        )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)


def get_risk_category(probability: float):
    if probability < 0.3:
        return "Low Risk", "#2e7d32"
    if probability <= 0.7:
        return "Moderate Risk", "#f9a825"
    return "High Risk", "#c62828"


def get_explanations(disease: str, x: dict, probability: float):
    factors = []

    if disease == "Diabetes":
        if x.get("glucose", 0) > 140:
            factors.append("High glucose level increased risk.")
        if x.get("bmi", 0) >= 30:
            factors.append("Higher BMI contributed to elevated risk.")
        if x.get("blood_pressure", 0) >= 140:
            factors.append("Elevated blood pressure contributed to risk.")
        if x.get("smoking") == "Yes":
            factors.append("Smoking status increased risk.")
        if x.get("physical_activity") == "Low":
            factors.append("Low physical activity contributed.")

    elif disease == "Heart Disease":
        if x.get("Cholesterol", 0) >= 240:
            factors.append("High cholesterol increased risk.")
        if x.get("RestingBP", 0) >= 140:
            factors.append("High resting blood pressure contributed.")
        if x.get("ExerciseAngina") == "Yes":
            factors.append("Exercise-induced angina increased risk.")
        if x.get("Oldpeak", 0) >= 1.5:
            factors.append("Abnormal ST depression (Oldpeak) contributed.")
        if x.get("ChestPainType") == "ASY":
            factors.append("Asymptomatic chest pain pattern increased risk.")

    elif disease == "COPD":
        if x.get("SmokingYears", 0) >= 10:
            factors.append("Long smoking history increased risk.")
        if x.get("BreathlessnessLevel", 1) >= 3:
            factors.append("Higher breathlessness level contributed.")
        if x.get("ChronicCough") == "Yes":
            factors.append("Chronic cough is a key risk indicator.")
        if x.get("AirPollutionExposure") == "High":
            factors.append("High air pollution exposure contributed.")

    elif disease == "Chronic Kidney Disease":
        if x.get("SerumCreatinine", 0) > 1.3:
            factors.append("Elevated serum creatinine increased risk.")
        if x.get("BloodUrea", 0) > 40:
            factors.append("High blood urea contributed to risk.")
        if x.get("Hemoglobin", 20) < 12:
            factors.append("Lower hemoglobin contributed.")
        if x.get("PedalEdema") == "Yes":
            factors.append("Pedal edema increased concern.")
        if x.get("Appetite") == "Poor":
            factors.append("Poor appetite contributed to risk profile.")

    if not factors:
        if probability > 0.7:
            factors.append("Multiple combined parameters increased risk.")
        else:
            factors.append("Current input pattern indicates relatively stable risk.")

    return factors[:3]


def get_recommendations(category: str):
    if category == "Low Risk":
        return [
            "Maintain healthy lifestyle habits.",
            "Continue regular physical activity.",
            "Monitor key parameters periodically.",
        ]
    if category == "Moderate Risk":
        return [
            "Consult a healthcare professional for further evaluation.",
            "Improve diet and exercise consistency.",
            "Track key parameters weekly.",
        ]
    return [
        "Consider medical consultation at the earliest.",
        "Monitor key parameters regularly and follow-up promptly.",
        "Adopt strict lifestyle control (diet, activity, smoking cessation).",
    ]


if submitted:
    input_data = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
        "smoking": smoking,
        "physical_activity": physical_activity,
        **disease_inputs,
    }

    disease_key = disease_key_map[disease_label]

    if disease_key == "diabetes":
        input_data.update(
            {
                "blood_glucose_level": float(glucose),
                "smoking_history": "current" if smoking == "Yes" else "never",
                "hypertension": 1 if blood_pressure >= 140 else 0,
                "heart_disease": 0,
                "HbA1c_level": round(max(4.0, min(14.0, 4 + glucose / 40)), 1),
            }
        )

    if disease_key == "heart":
        input_data.update(
            {
                "Age": age,
                "FastingBS": 1 if glucose >= 120 else 0,
                "Sex_M": 1 if gender == "Male" else 0,
                "ExerciseAngina_Y": 1 if disease_inputs.get("ExerciseAngina") == "Yes" else 0,
                "ChestPainType_ATA": 1 if disease_inputs.get("ChestPainType") == "ATA" else 0,
                "ChestPainType_NAP": 1 if disease_inputs.get("ChestPainType") == "NAP" else 0,
                "ChestPainType_TA": 1 if disease_inputs.get("ChestPainType") == "TA" else 0,
                "RestingECG_Normal": 1 if disease_inputs.get("RestingECG") == "Normal" else 0,
                "RestingECG_ST": 1 if disease_inputs.get("RestingECG") == "ST" else 0,
                "ST_Slope_Flat": 1 if disease_inputs.get("ST_Slope") == "Flat" else 0,
                "ST_Slope_Up": 1 if disease_inputs.get("ST_Slope") == "Up" else 0,
            }
        )

    missing = [k for k, v in input_data.items() if v is None or (isinstance(v, str) and not v.strip())]
    if missing:
        st.warning("Please fill all required inputs before prediction.")
    else:
        try:
            result = predict_risk(disease_key, input_data)
            probability = float(result.get("probability", 0.0))
            risk_category, risk_color = get_risk_category(probability)
            score_text = f"{round(probability * 100)}%"

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### Prediction Result")

            st.markdown(
                f"""
                <div class="risk-card">
                    <p class="risk-score" style="color:{risk_color};">{score_text}</p>
                    <p class="risk-label" style="color:{risk_color};">{risk_category}</p>
                    <p style="margin-top:0.4rem;">{result.get("label", "")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(max(probability, 0.0), 1.0))

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### Explanation")
            for item in get_explanations(disease_label, input_data, probability):
                st.write(f"- {item}")

            st.markdown("### Recommendations")
            for rec in get_recommendations(risk_category):
                st.write(f"- {rec}")

        except FileNotFoundError:
            st.warning(
                f"Model artifacts for '{disease_label}' are not available yet. "
                "Train and save model/scaler/threshold first."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
