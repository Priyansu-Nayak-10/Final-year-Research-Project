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

    .stButton > button[kind="primary"] {
        background: #16a34a !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #15803d !important;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background: #15803d !important;
        color: #ffffff !important;
    }

    /* Reset / Clear Form button (red) */
    .stButton > button[kind="secondary"] {
        background: #dc2626 !important;
        color: #ffffff !important;
        border: 1px solid #b91c1c !important;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #b91c1c !important;
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
    "diabetes": ["diabetes_model.pkl"],
    "cardiovascular": ["cardiovascular_model.pkl"],
}

DATASET_FILES = {
    "diabetes": "diabetes.csv",
    "cardiovascular": "cardiovascular.csv",
}


FEATURE_UI_CONFIG = {
    "age": {
        "label": "Age",
        "type": "number",
        "min": 18.0,
        "max": 90.0,
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
        "min": 15.0,
        "max": 50.0,
        "step": 0.1,
        "placeholder": "e.g., 24.7",
    },
    "blood_pressure": {
        "label": "Blood Pressure (mmHg)",
        "type": "number",
        "min": 90.0,
        "max": 200.0,
        "step": 1.0,
        "placeholder": "e.g., 120",
    },
    "fasting_glucose_level": {
        "label": "Fasting Glucose (mg/dL)",
        "type": "number",
        "min": 60.0,
        "max": 300.0,
        "step": 1.0,
        "placeholder": "e.g., 95",
    },
    "insulin_level": {
        "label": "Insulin Level (uIU/mL)",
        "type": "number",
        "min": 2.0,
        "max": 60.0,
        "step": 0.1,
        "placeholder": "e.g., 10.0",
    },
    "HbA1c_level": {
        "label": "HbA1c Level (%)",
        "type": "number",
        "min": 4.0,
        "max": 12.0,
        "step": 0.1,
        "placeholder": "e.g., 5.6",
    },
    "cholesterol_level": {
        "label": "Cholesterol Level (mg/dL)",
        "type": "number",
        "min": 120.0,
        "max": 350.0,
        "step": 1.0,
        "placeholder": "e.g., 185",
    },
    "triglycerides_level": {
        "label": "Triglycerides (mg/dL)",
        "type": "number",
        "min": 50.0,
        "max": 500.0,
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
        "min": 1200.0,
        "max": 5500.0,
        "step": 10.0,
        "placeholder": "e.g., 2200",
    },
    "sugar_intake_grams_per_day": {
        "label": "Sugar Intake (grams/day)",
        "type": "number",
        "min": 0.0,
        "max": 300.0,
        "step": 0.5,
        "placeholder": "e.g., 45",
    },
    "sleep_hours": {
        "label": "Sleep Hours (per day)",
        "type": "number",
        "min": 4.0,
        "max": 10.0,
        "step": 0.1,
        "placeholder": "e.g., 7.0",
    },
    "stress_level": {
        "label": "Stress Level (0-10)",
        "type": "number",
        "min": 1.0,
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
        "min": 60.0,
        "max": 150.0,
        "step": 0.1,
        "placeholder": "e.g., 92",
    },
    "systolic_bp": {
        "label": "Systolic BP (mmHg)",
        "type": "number",
        "min": 90.0,
        "max": 220.0,
        "step": 1.0,
        "placeholder": "e.g., 120",
    },
    "diastolic_bp": {
        "label": "Diastolic BP (mmHg)",
        "type": "number",
        "min": 50.0,
        "max": 130.0,
        "step": 1.0,
        "placeholder": "e.g., 80",
    },
    "cholesterol_mg_dl": {
        "label": "Cholesterol (mg/dL)",
        "type": "number",
        "min": 120.0,
        "max": 350.0,
        "step": 1.0,
        "placeholder": "e.g., 190",
    },
    "resting_heart_rate": {
        "label": "Resting Heart Rate (bpm)",
        "type": "number",
        "min": 40.0,
        "max": 120.0,
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
        "min": 500.0,
        "max": 20000.0,
        "step": 100.0,
        "placeholder": "e.g., 7000",
    },
    "physical_activity_hours_per_week": {
        "label": "Physical Activity (hours/week)",
        "type": "number",
        "min": 0.0,
        "max": 14.0,
        "step": 0.1,
        "placeholder": "e.g., 3.5",
    },
    "family_history_heart_disease": {
        "label": "Family History of Cardiovascular Disease",
        "type": "select",
        "options": ["No", "Yes"],
    },
    "diet_quality_score": {
        "label": "Diet Quality Score (0-10)",
        "type": "number",
        "min": 1.0,
        "max": 10.0,
        "step": 0.1,
        "placeholder": "e.g., 6.5",
    },
    "alcohol_units_per_week": {
        "label": "Alcohol Units / Week",
        "type": "number",
        "min": 0.0,
        "max": 30.0,
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


def _resolve_existing_path_candidates(directory: Path, filenames: List[str], label: str) -> Path:
    for filename in filenames:
        candidate = (directory / filename).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} not found. Checked: {', '.join(str((directory / name).resolve()) for name in filenames)}")


@st.cache_resource(show_spinner=False)
def load_model_bundle(disease_key: str) -> Dict[str, Any]:
    """Load the main model artifact and compatible optional component files."""
    model_path = _resolve_existing_path_candidates(MODEL_DIR, MODEL_FILES[disease_key], "Model artifact")
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

    # Only load the primary artifact; do not attempt to load optional standalone components.

    if bundle["model"] is None or bundle["preprocessor"] is None:
        raise ValueError(f"Incomplete model bundle for '{disease_key}': missing model or preprocessor.")

    return bundle


# Helper Functions
@st.cache_data(show_spinner=False)
def load_dataset(disease_key: str) -> pd.DataFrame:
    dataset_path = _resolve_existing_path(DATA_DIR, DATASET_FILES[disease_key], "Dataset")
    return pd.read_csv(dataset_path)




def render_model_info_card(disease_key: str, disease_label: str) -> None:
    try:
        bundle = load_model_bundle(disease_key)
        dataset = load_dataset(disease_key)
        feature_list = bundle.get("features", bundle.get("numeric_cols", []) + bundle.get("categorical_cols", []))
        selected_features = bundle.get("selected_features", [])
        numeric_cols = bundle.get("numeric_cols", [])
        categorical_cols = bundle.get("categorical_cols", [])

        st.markdown(
            f"""
            <div class="section-card">
                <h4 class="soft-title" style="margin-bottom:0.3rem;">{disease_label} Model</h4>
                <p class="small-note" style="margin:0.15rem 0;"><b>Algorithm:</b> {bundle.get('model_name', 'Trained Model')}</p>
                <p class="small-note" style="margin:0.15rem 0;"><b>Dataset Records:</b> {len(dataset):,}</p>
                <p class="small-note" style="margin:0.15rem 0;"><b>Total Input Features:</b> {len(feature_list)}</p>
                <p class="small-note" style="margin:0.15rem 0;"><b>Selected Features (SelectKBest):</b> {len(selected_features) if selected_features else len(feature_list)}</p>
                <p class="small-note" style="margin:0.15rem 0;"><b>Decision Threshold:</b> {float(bundle.get('threshold', 0.5)):.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"{disease_label} feature details"):
            st.write(f"Numeric Features ({len(numeric_cols)}): " + ", ".join(numeric_cols))
            st.write(f"Categorical Features ({len(categorical_cols)}): " + ", ".join(categorical_cols))
            if selected_features:
                st.write(f"Selected Features ({len(selected_features)}): " + ", ".join([str(f) for f in selected_features]))
    except Exception as exc:
        st.error(f"Unable to load {disease_label.lower()} model information: {exc}")


def is_missing(value: Any) -> bool:
    return value is None or value == "Select"


def get_form_version(page_key: str) -> int:
    return int(st.session_state.get(f"{page_key}_form_version", 0))


def build_input_form(feature_list: List[str], form_key: str) -> Dict[str, Any]:
    """Render a dynamic input form using selected raw features only."""
    user_values: Dict[str, Any] = {}
    form_version = get_form_version(form_key)

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
                    key=f"{form_key}_{feature}_v{form_version}",
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
                    key=f"{form_key}_{feature}_v{form_version}",
                )
            else:
                options = ["Select"] + cfg["options"]
                user_values[feature] = st.selectbox(
                    cfg["label"],
                    options=options,
                    index=0,
                    key=f"{form_key}_{feature}_v{form_version}",
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


def render_result_card(prediction: int, disease_name: str) -> None:
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
            <p class="small-note" style="margin-top:0.45rem;">{interpretation_map[risk_level]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### AI-Based Health Recommendations")
    for recommendation in get_health_recommendations(risk_level, disease_name):
        st.write(f"- {recommendation}")


def prediction_state_key(disease_key: str) -> str:
    return f"prediction_result_{disease_key}"


def save_prediction_result(disease_key: str, prediction: int, probability: float, threshold: float) -> None:
    st.session_state[prediction_state_key(disease_key)] = {
        "prediction": int(prediction),
        "probability": float(probability),
        "threshold": float(threshold),
    }


def render_saved_prediction_result(disease_key: str, disease_name: str) -> None:
    saved_result = st.session_state.get(prediction_state_key(disease_key))
    if not saved_result:
        return

    render_result_card(int(saved_result["prediction"]), disease_name)


def clear_prediction_page_state(page_key: str) -> None:
    version_key = f"{page_key}_form_version"
    current_version = int(st.session_state.get(version_key, 0))
    for key in list(st.session_state.keys()):
        if key == version_key:
            continue
        if key.startswith(f"{page_key}_") or key == prediction_state_key(page_key):
            del st.session_state[key]
    st.session_state[version_key] = current_version + 1


# Sidebar Navigation
st.sidebar.markdown("## Navigation")
selected_page = st.sidebar.radio(
        "Go to",
        [
        "Home",
        "Diabetes Prediction",
        "Cardiovascular Prediction",
        "Model Information",
        ],
)

st.sidebar.markdown("---")


# Home Page
if selected_page == "Home":
    st.markdown(
        f"""
        <div class="hero-card">
            <h2 class="soft-title">{APP_TITLE}</h2>
            <p class="muted-text">
                Welcome to the AI healthcare dashboard for early <b>Diabetes</b> and <b>Cardiovascular Disease</b> risk prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    btn_col, reset_col = st.columns([3, 1])
    with btn_col:
        predict_clicked = st.button("Predict Diabetes Risk", use_container_width=True, type="primary")
    with reset_col:
        if st.button("Clear Inputs", key="reset_diabetes", use_container_width=True, type="secondary"):
            clear_prediction_page_state("diabetes")
            st.rerun()

    if predict_clicked:
        valid, missing = validate_inputs(diabetes_inputs, diabetes_all_features)
        if not valid:
            st.error("Please fill required fields: " + ", ".join(missing))
        else:
            with st.spinner("Running diabetes risk prediction..."):
                try:
                    pred, probability, threshold = run_prediction_pipeline(diabetes_inputs, diabetes_bundle)
                    save_prediction_result("diabetes", pred, probability, threshold)
                    st.success("Prediction completed successfully.")
                except Exception as exc:
                    st.error(f"Prediction failed. Please check inputs/artifacts. Error: {exc}")

    render_saved_prediction_result("diabetes", "Diabetes")


# Cardiovascular Prediction Page
elif selected_page == "Cardiovascular Prediction":
    st.title("Cardiovascular Disease Risk Prediction")

    try:
        cardiovascular_bundle = load_model_bundle("cardiovascular")
    except Exception as exc:
        st.error(f"Unable to load cardiovascular model artifacts: {exc}")
        st.stop()

    cardiovascular_all_features = cardiovascular_bundle.get(
        "features",
        cardiovascular_bundle.get("numeric_cols", []) + cardiovascular_bundle.get("categorical_cols", []),
    )

    cardiovascular_inputs = build_input_form(
        feature_list=cardiovascular_all_features,
        form_key="cardiovascular",
    )

    btn_col, reset_col = st.columns([3, 1])
    with btn_col:
        predict_clicked = st.button("Predict Cardiovascular Risk", use_container_width=True, type="primary")
    with reset_col:
        if st.button("Clear Inputs", key="reset_cardiovascular", use_container_width=True, type="secondary"):
            clear_prediction_page_state("cardiovascular")
            st.rerun()

    if predict_clicked:
        valid, missing = validate_inputs(cardiovascular_inputs, cardiovascular_all_features)
        if not valid:
            st.error("Please fill required fields: " + ", ".join(missing))
        else:
            with st.spinner("Running cardiovascular risk prediction..."):
                try:
                    pred, probability, threshold = run_prediction_pipeline(cardiovascular_inputs, cardiovascular_bundle)
                    save_prediction_result("cardiovascular", pred, probability, threshold)
                    st.success("Prediction completed successfully.")
                except Exception as exc:
                    st.error(f"Prediction failed. Please check inputs/artifacts. Error: {exc}")

    render_saved_prediction_result("cardiovascular", "Cardiovascular Disease")


# Model Information Page
elif selected_page == "Model Information":
    st.title("Model Information")
    st.markdown(
        """
        <div class="section-card">
            <p class="muted-text" style="margin:0;">
                This section provides model and dataset details for both prediction modules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)
    with left_col:
        render_model_info_card("diabetes", "Diabetes")
    with right_col:
        render_model_info_card("cardiovascular", "Cardiovascular Disease")


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


