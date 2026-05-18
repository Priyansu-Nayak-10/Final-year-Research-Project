import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Imports + Paths
def resolve_project_root() -> Path:
    """
    Resolve project root safely across local/dev/deployment environments.
    We walk upward from this file and choose the best-matching candidate.
    `models/` is treated as required and other known directories increase confidence.
    """
    current_file = Path(__file__).resolve()
    known_dirs = ("models", "data", "Files", "app")
    best_candidate = None
    best_score = -1

    for candidate in [current_file.parent, *current_file.parents]:
        if not (candidate / "models").exists():
            continue
        score = sum(1 for directory in known_dirs if (candidate / directory).exists())
        if score > best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate is not None:
        return best_candidate

    # Final safe fallback for unusual packaging layouts.
    return current_file.parent


BASE_DIR = resolve_project_root()
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "Files"


# Page Configuration
st.set_page_config(
    page_title="AI-Based Early Disease Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS Styling (White + Green Medical Theme)
st.markdown(
    """
    <style>
    :root {
        --primary: #1f9d55;
        --primary-soft: #e8f8ef;
        --accent: #f5f7f9;
        --text-dark: #1f2937;
        --muted: #6b7280;
        --danger: #dc2626;
        --success: #059669;
    }

    .stApp {
        background: #ffffff;
        color: var(--text-dark);
    }

    .hero-card,
    .section-card,
    .result-card,
    .stat-card,
    .footer-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 6px 24px rgba(16, 24, 40, 0.06);
    }

    .hero-card {
        border-left: 6px solid var(--primary);
        background: linear-gradient(120deg, #ffffff 0%, #f1fbf5 100%);
        margin-bottom: 1rem;
    }

    .soft-title {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .muted-text {
        color: var(--muted);
        font-size: 0.95rem;
    }

    .risk-low {
        border-left: 6px solid var(--success);
        background: #ecfdf5;
    }

    .risk-high {
        border-left: 6px solid var(--danger);
        background: #fef2f2;
    }

    .small-note {
        font-size: 0.88rem;
        color: #4b5563;
    }

    .stButton > button {
        background: #16a34a !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #15803d !important;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #15803d !important;
        color: #ffffff !important;
    }

    /* Number input +/- controls in green */
    div[data-testid="stNumberInput"] button {
        background: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #15803d !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #15803d !important;
        color: #ffffff !important;
    }

    /* Make all input labels and values clearly visible in black */
    label[data-testid="stWidgetLabel"] p,
    .stMarkdown,
    .stText,
    .stCaption {
        color: #111111 !important;
    }

    div[data-baseweb="input"] input {
        color: #111111 !important;
        background: #ffffff !important;
    }

    /* Dropdown/select white theme */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #111111 !important;
        border-color: #d1d5db !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg {
        color: #111111 !important;
        fill: #111111 !important;
    }

    ul[role="listbox"] {
        background: #ffffff !important;
        color: #111111 !important;
    }

    ul[role="listbox"] li {
        background: #ffffff !important;
        color: #111111 !important;
    }

    ul[role="listbox"] li:hover {
        background: #f3f4f6 !important;
    }

    /* Sidebar navigation text in white */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Constants + Metadata
APP_TITLE = "AI-Based Early Disease Risk Prediction System Using Machine Learning"

MODEL_FILES = {
    "diabetes": "diabetes_model.pkl",
    "heart": "heart_model.pkl",
}

DATASET_FILES = {
    "diabetes": "diabetes.csv",
    "heart": "cardiovascular.csv",
}

ASSET_FILES = {
    "model_architecture": "model architecture.png",
    "model_methodology": "Model Methodology.png",
}

FEATURE_UI_CONFIG = {
    "age": {
        "label": "Age",
        "type": "number",
        "min": 1.0,
        "max": 100.0,
        "step": 1.0,
        "placeholder": "e.g., 45",
    },
    "gender": {
        "label": "Gender",
        "type": "select",
        "options": ["Female", "Male"],
    },
    "bmi": {
        "label": "Body Mass Index (BMI)",
        "type": "number",
        "min": 10.0,
        "max": 60.0,
        "step": 0.1,
        "placeholder": "e.g., 24.7",
    },
    "blood_pressure": {
        "label": "Blood Pressure (mmHg)",
        "type": "number",
        "min": 60.0,
        "max": 240.0,
        "step": 1.0,
        "placeholder": "e.g., 120",
    },
    "fasting_glucose_level": {
        "label": "Fasting Glucose (mg/dL)",
        "type": "number",
        "min": 50.0,
        "max": 350.0,
        "step": 1.0,
        "placeholder": "e.g., 95",
    },
    "insulin_level": {
        "label": "Insulin Level",
        "type": "number",
        "min": 0.0,
        "max": 400.0,
        "step": 0.1,
        "placeholder": "e.g., 15.5",
    },
    "HbA1c_level": {
        "label": "HbA1c Level (%)",
        "type": "number",
        "min": 3.0,
        "max": 20.0,
        "step": 0.1,
        "placeholder": "e.g., 5.6",
    },
    "cholesterol_level": {
        "label": "Cholesterol Level (mg/dL)",
        "type": "number",
        "min": 80.0,
        "max": 500.0,
        "step": 1.0,
        "placeholder": "e.g., 185",
    },
    "triglycerides_level": {
        "label": "Triglycerides (mg/dL)",
        "type": "number",
        "min": 50.0,
        "max": 700.0,
        "step": 1.0,
        "placeholder": "e.g., 150",
    },
    "physical_activity_level": {
        "label": "Physical Activity Level",
        "type": "select",
        "options": ["Low", "Moderate", "High"],
    },
    "daily_calorie_intake": {
        "label": "Daily Calorie Intake",
        "type": "number",
        "min": 500.0,
        "max": 6000.0,
        "step": 10.0,
        "placeholder": "e.g., 2200",
    },
    "sugar_intake_grams_per_day": {
        "label": "Sugar Intake (grams/day)",
        "type": "number",
        "min": 0.0,
        "max": 500.0,
        "step": 0.5,
        "placeholder": "e.g., 45",
    },
    "sleep_hours": {
        "label": "Sleep Hours (per day)",
        "type": "number",
        "min": 0.0,
        "max": 14.0,
        "step": 0.1,
        "placeholder": "e.g., 7.0",
    },
    "stress_level": {
        "label": "Stress Level (0-10)",
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "placeholder": "e.g., 5",
    },
    "family_history_diabetes": {
        "label": "Family History of Diabetes",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "waist_circumference_cm": {
        "label": "Waist Circumference (cm)",
        "type": "number",
        "min": 40.0,
        "max": 200.0,
        "step": 0.1,
        "placeholder": "e.g., 92",
    },
    "systolic_bp": {
        "label": "Systolic BP (mmHg)",
        "type": "number",
        "min": 70.0,
        "max": 260.0,
        "step": 1.0,
        "placeholder": "e.g., 120",
    },
    "diastolic_bp": {
        "label": "Diastolic BP (mmHg)",
        "type": "number",
        "min": 40.0,
        "max": 160.0,
        "step": 1.0,
        "placeholder": "e.g., 80",
    },
    "cholesterol_mg_dl": {
        "label": "Cholesterol (mg/dL)",
        "type": "number",
        "min": 80.0,
        "max": 500.0,
        "step": 1.0,
        "placeholder": "e.g., 190",
    },
    "resting_heart_rate": {
        "label": "Resting Heart Rate (bpm)",
        "type": "number",
        "min": 35.0,
        "max": 220.0,
        "step": 1.0,
        "placeholder": "e.g., 72",
    },
    "smoking_status": {
        "label": "Smoking Status",
        "type": "select",
        "options": ["Never", "Former", "Current"],
    },
    "daily_steps": {
        "label": "Daily Steps",
        "type": "number",
        "min": 0.0,
        "max": 60000.0,
        "step": 100.0,
        "placeholder": "e.g., 7000",
    },
    "physical_activity_hours_per_week": {
        "label": "Physical Activity (hours/week)",
        "type": "number",
        "min": 0.0,
        "max": 60.0,
        "step": 0.1,
        "placeholder": "e.g., 3.5",
    },
    "family_history_heart_disease": {
        "label": "Family History of Heart Disease",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "diet_quality_score": {
        "label": "Diet Quality Score (0-10)",
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "placeholder": "e.g., 6.5",
    },
    "alcohol_units_per_week": {
        "label": "Alcohol Units / Week",
        "type": "number",
        "min": 0.0,
        "max": 80.0,
        "step": 0.5,
        "placeholder": "e.g., 2",
    },
}

# Load Models Section
def _resolve_existing_path(directory: Path, filename: str, label: str) -> Path:
    path = (directory / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")
    return path


@st.cache_resource(show_spinner=False)
def load_model_bundle(disease_key: str) -> Dict[str, Any]:
    """Load the main model artifact and compatible optional component files."""
    model_path = _resolve_existing_path(MODEL_DIR, MODEL_FILES[disease_key], "Model artifact")
    artifact = joblib.load(model_path)

    bundle = {
        "model": artifact.get("model"),
        "preprocessor": artifact.get("preprocessor"),
        "selector": artifact.get("selector"),
        "features": artifact.get("features", []),
        "selected_features": artifact.get("selected_features", []),
        "numeric_cols": artifact.get("numeric_cols", []),
        "categorical_cols": artifact.get("categorical_cols", []),
        "threshold": artifact.get("threshold", 0.5),
        "model_name": artifact.get("model_name", "Trained Model"),
    }

    # Optional standalone files (if present in future versions of the project)
    for key in ["features", "selector", "selected_features"]:
        component_path = MODEL_DIR / f"{disease_key}_{key}.pkl"
        if component_path.exists():
            bundle[key] = joblib.load(component_path)

    if bundle["model"] is None or bundle["preprocessor"] is None:
        raise ValueError(f"Incomplete model bundle for '{disease_key}': missing model or preprocessor.")

    return bundle


# Helper Functions
@st.cache_data(show_spinner=False)
def load_dataset(disease_key: str) -> pd.DataFrame:
    dataset_path = _resolve_existing_path(DATA_DIR, DATASET_FILES[disease_key], "Dataset")
    return pd.read_csv(dataset_path)


def get_asset_path(asset_key: str) -> Path:
    return _resolve_existing_path(ASSETS_DIR, ASSET_FILES[asset_key], "Asset")


def is_missing(value: Any) -> bool:
    return value is None or value == "Select"


def build_input_form(feature_list: List[str], form_key: str) -> Dict[str, Any]:
    """Render a dynamic input form using selected raw features only."""
    user_values: Dict[str, Any] = {}

    columns = st.columns(4)
    for idx, feature in enumerate(feature_list):
        col = columns[idx % 4]
        cfg = FEATURE_UI_CONFIG.get(feature)

        if cfg is None:
            # Generic fallback for unknown numeric-like feature.
            with col:
                user_values[feature] = st.number_input(
                    f"{feature}",
                    value=None,
                    placeholder="Enter value",
                    key=f"{form_key}_{feature}",
                )
            continue

        with col:
            if cfg["type"] == "number":
                label_with_range = f"{cfg['label']} [{cfg['min']} - {cfg['max']}]"
                user_values[feature] = st.number_input(
                    label_with_range,
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    step=float(cfg["step"]),
                    value=None,
                    placeholder=cfg["placeholder"],
                    key=f"{form_key}_{feature}",
                )
            else:
                options = ["Select"] + cfg["options"]
                user_values[feature] = st.selectbox(
                    cfg["label"],
                    options=options,
                    index=0,
                    key=f"{form_key}_{feature}",
                )

    return user_values


def validate_inputs(inputs: Dict[str, Any], required_features: List[str]) -> Tuple[bool, List[str]]:
    """Simple validation for required selected fields."""
    missing_labels = []
    for feature in required_features:
        value = inputs.get(feature)
        if is_missing(value):
            label = FEATURE_UI_CONFIG.get(feature, {}).get("label", feature)
            missing_labels.append(label)
    return len(missing_labels) == 0, missing_labels


def run_prediction_pipeline(inputs: Dict[str, Any], bundle: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Keep prediction flow aligned with training pipeline:
    1) Encoding + scaling by preprocessor
    2) Feature selection by SelectKBest
    3) Prediction + probability
    """
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    selector = bundle["selector"]
    selected_features = bundle.get("selected_features", [])
    threshold = float(bundle.get("threshold", 0.5))

    input_cols = bundle.get("features", bundle.get("numeric_cols", []) + bundle.get("categorical_cols", []))
    input_df = pd.DataFrame([inputs])

    # Step 1: Encode categorical + scale numerical features.
    processed = preprocessor.transform(input_df[input_cols])

    # Step 2: Keep top-k selected features (same as training).
    if selector is not None:
        selected_data = selector.transform(processed)
    elif selected_features:
        # Fallback path for future artifacts missing explicit selector object.
        try:
            processed_cols = list(preprocessor.get_feature_names_out(input_cols))
            processed_df = pd.DataFrame(processed, columns=processed_cols)
            selected_data = processed_df[selected_features].to_numpy()
        except Exception:
            selected_data = processed
    else:
        selected_data = processed

    # Step 3: Predict class and disease probability.
    probability = float(model.predict_proba(selected_data)[0][1])
    prediction = 1 if probability >= threshold else 0

    return prediction, probability, threshold


def risk_level_from_prediction(prediction: int) -> str:
    """Convert binary model prediction to risk level label."""
    return "High Risk" if prediction == 1 else "Low Risk"


def risk_class_for_css(risk_level: str) -> str:
    if risk_level == "Low Risk":
        return "risk-low"
    return "risk-high"


def get_health_recommendations(risk_level: str, disease_name: str) -> List[str]:
    if risk_level == "High Risk":
        return [
            f"Consult a physician soon for detailed {disease_name.lower()} screening.",
            "Follow a low-sugar, low-saturated-fat diet plan.",
            "Start regular exercise (at least 30 minutes/day, 5 days/week).",
            "Track blood markers regularly and reduce lifestyle stress.",
        ]

    return [
        "Continue a balanced diet and active routine.",
        "Maintain healthy body weight and stress management.",
        "Do preventive health checkups at regular intervals.",
        "Avoid smoking and limit processed sugar intake.",
    ]


def render_result_card(prediction: int, probability: float, disease_name: str, threshold: float) -> None:
    risk_level = risk_level_from_prediction(prediction)
    css_class = risk_class_for_css(risk_level)

    interpretation_map = {
        "Low Risk": f"Current model indicates relatively low {disease_name.lower()} risk.",
        "High Risk": f"Current model indicates high {disease_name.lower()} risk. Clinical consultation is strongly recommended.",
    }

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <h4 style="margin-bottom:0.25rem;">Prediction Result: {risk_level}</h4>
            <p style="margin:0.15rem 0;">Risk Probability: <b>{probability * 100:.2f}%</b></p>
            <p style="margin:0.15rem 0;">Prediction Confidence: <b>{max(probability, 1-probability) * 100:.2f}%</b></p>
            <p style="margin:0.15rem 0;">Decision Threshold Used: <b>{threshold:.2f}</b></p>
            <p class="small-note" style="margin-top:0.45rem;">{interpretation_map[risk_level]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### AI-Based Health Recommendations")
    for recommendation in get_health_recommendations(risk_level, disease_name):
        st.write(f"- {recommendation}")


# Sidebar Navigation
st.sidebar.markdown("## Navigation")
selected_page = st.sidebar.radio(
        "Go to",
        [
        "Home",
        "Diabetes Prediction",
        "Heart Disease Prediction",
        ],
)

st.sidebar.markdown("---")
st.sidebar.info("This app is for academic/research support only, not a final medical diagnosis.")


# Home Page
if selected_page == "Home":
    st.markdown(
        f"""
        <div class="hero-card">
            <h2 class="soft-title">{APP_TITLE}</h2>
            <p class="muted-text">
                A research-oriented AI healthcare dashboard for early <b>Diabetes</b> and <b>Heart Disease</b> risk prediction.
                The system demonstrates ML-based preventive screening support with interpretable probability outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='stat-card'><h4>2</h4><p class='small-note'>Risk Prediction Modules</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='stat-card'><h4>5+</h4><p class='small-note'>Algorithms Evaluated</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='stat-card'><h4>SelectKBest</h4><p class='small-note'>Feature Selection</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='stat-card'><h4>Threshold Tuning</h4><p class='small-note'>Optimized Decisions</p></div>", unsafe_allow_html=True)

    st.markdown("### Why Early Disease Prediction Matters")
    st.write(
        "Early detection helps reduce complications, supports timely interventions, and enables personalized preventive care."
    )

    st.markdown("### ML Technologies Used")
    st.write("- Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost")
    st.write("- Data preprocessing with encoding and scaling")
    st.write("- SMOTE for class balancing")
    st.write("- SelectKBest for feature selection")
    st.write("- Cross-validation and threshold optimization")


# Diabetes Prediction Page
elif selected_page == "Diabetes Prediction":
    st.title("Diabetes Risk Prediction")

    try:
        diabetes_bundle = load_model_bundle("diabetes")
    except Exception as exc:
        st.error(f"Unable to load diabetes model artifacts: {exc}")
        st.stop()

    diabetes_all_features = diabetes_bundle.get(
        "features",
        diabetes_bundle.get("numeric_cols", []) + diabetes_bundle.get("categorical_cols", []),
    )

    diabetes_inputs = build_input_form(
        feature_list=diabetes_all_features,
        form_key="diabetes",
    )

    if st.button("Predict Diabetes Risk", use_container_width=True):
        valid, missing = validate_inputs(diabetes_inputs, diabetes_all_features)
        if not valid:
            st.error("Please fill required fields: " + ", ".join(missing))
        else:
            with st.spinner("Running diabetes risk prediction..."):
                try:
                    pred, prob, threshold = run_prediction_pipeline(diabetes_inputs, diabetes_bundle)
                    st.success("Prediction completed successfully.")
                    render_result_card(pred, prob, "Diabetes", threshold)
                except Exception as exc:
                    st.error(f"Prediction failed. Please check inputs/artifacts. Error: {exc}")


# Heart Disease Prediction Page
elif selected_page == "Heart Disease Prediction":
    st.title("Heart Disease Risk Prediction")

    try:
        heart_bundle = load_model_bundle("heart")
    except Exception as exc:
        st.error(f"Unable to load heart model artifacts: {exc}")
        st.stop()

    heart_all_features = heart_bundle.get(
        "features",
        heart_bundle.get("numeric_cols", []) + heart_bundle.get("categorical_cols", []),
    )

    heart_inputs = build_input_form(
        feature_list=heart_all_features,
        form_key="heart",
    )

    if st.button("Predict Heart Disease Risk", use_container_width=True):
        valid, missing = validate_inputs(heart_inputs, heart_all_features)
        if not valid:
            st.error("Please fill required fields: " + ", ".join(missing))
        else:
            with st.spinner("Running heart disease risk prediction..."):
                try:
                    pred, prob, threshold = run_prediction_pipeline(heart_inputs, heart_bundle)
                    st.success("Prediction completed successfully.")
                    render_result_card(pred, prob, "Heart Disease", threshold)
                except Exception as exc:
                    st.error(f"Prediction failed. Please check inputs/artifacts. Error: {exc}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer-card">
        <p style="margin:0; text-align:center; color:#4b5563;">
            Final Year Research Project • AI-Based Early Disease Risk Prediction System • Streamlit + Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


