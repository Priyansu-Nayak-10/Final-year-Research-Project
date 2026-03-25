from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
_SCALED_MODEL_CLASS_NAMES = {
    "LogisticRegression",
    "SVC",
    "LinearSVC",
    "SGDClassifier",
    "MLPClassifier",
    "KNeighborsClassifier",
}

_SMOKING_TO_COLUMN = {
    "current": "smoking_history_current",
    "ever": "smoking_history_ever",
    "former": "smoking_history_former",
    "never": "smoking_history_never",
    "not current": "smoking_history_not current",
    "not_current": "smoking_history_not current",
    "not-current": "smoking_history_not current",
    "notcurrent": "smoking_history_not current",
}


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_binary(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value >= 1 else 0.0
    return 1.0 if _normalize_text(value) in {"1", "true", "yes", "y"} else 0.0


@lru_cache(maxsize=8)
def _load_artifacts(disease: str) -> tuple[Any, float, Any | None]:
    disease_name = _normalize_text(disease)
    disease_dir = MODELS_DIR / disease_name

    model_path = disease_dir / "model.pkl"
    scaler_path = disease_dir / "scaler.pkl"
    threshold_path = disease_dir / "threshold.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, threshold, scaler


def _infer_feature_order(model: Any, scaler: Any | None) -> list[str]:
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError(
        "Feature names are missing in model/scaler. Retrain and save artifacts with named columns."
    )


def _build_diabetes_features(user_input: Mapping[str, Any], feature_order: list[str]) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_order}

    if "age" in row:
        row["age"] = _to_float(user_input.get("age"))
    if "hypertension" in row:
        row["hypertension"] = _to_binary(user_input.get("hypertension"))
    if "heart_disease" in row:
        row["heart_disease"] = _to_binary(user_input.get("heart_disease"))
    if "bmi" in row:
        row["bmi"] = _to_float(user_input.get("bmi"))
    if "HbA1c_level" in row:
        row["HbA1c_level"] = _to_float(user_input.get("HbA1c_level"))
    if "blood_glucose_level" in row:
        row["blood_glucose_level"] = _to_float(user_input.get("blood_glucose_level"))

    gender = _normalize_text(user_input.get("gender", "female"))
    if gender in {"male", "m"} and "gender_Male" in row:
        row["gender_Male"] = 1.0
    elif gender in {"other", "o"} and "gender_Other" in row:
        row["gender_Other"] = 1.0

    smoking_history = _normalize_text(user_input.get("smoking_history", "not current"))
    smoking_column = _SMOKING_TO_COLUMN.get(smoking_history)
    if smoking_column and smoking_column in row:
        row[smoking_column] = 1.0

    return pd.DataFrame([row], columns=feature_order)


def _build_generic_features(user_input: Mapping[str, Any], feature_order: list[str]) -> pd.DataFrame:
    row = {col: _to_float(user_input.get(col, 0.0)) for col in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def _build_features(disease: str, user_input: Mapping[str, Any], feature_order: list[str]) -> pd.DataFrame:
    if _normalize_text(disease) == "diabetes":
        return _build_diabetes_features(user_input, feature_order)
    return _build_generic_features(user_input, feature_order)


def _requires_scaling(model: Any, scaler: Any | None) -> bool:
    if scaler is None:
        return False
    return model.__class__.__name__ in _SCALED_MODEL_CLASS_NAMES


def _label_for_prediction(disease: str, prediction: int) -> str:
    name = _normalize_text(disease)
    if prediction == 1:
        return f"High {name} risk"
    return f"Lower {name} risk"


def predict_risk(disease: str, user_input: Mapping[str, Any]) -> dict[str, Any]:
    model, threshold, scaler = _load_artifacts(disease)
    feature_order = _infer_feature_order(model, scaler)
    features_df = _build_features(disease, user_input, feature_order)

    model_input = features_df
    uses_scaler = _requires_scaling(model, scaler)
    if uses_scaler:
        scaled = scaler.transform(features_df)
        model_input = pd.DataFrame(scaled, columns=feature_order)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(model_input)[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))
    else:
        probability = float(model.predict(model_input)[0])

    prediction = int(probability >= threshold)
    disease_name = _normalize_text(disease)

    return {
        "disease": disease_name,
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "label": _label_for_prediction(disease_name, prediction),
        "model_name": model.__class__.__name__,
        "uses_scaler": uses_scaler,
        "model_input": features_df.iloc[0].to_dict(),
    }


def predict_diabetes(user_input: Mapping[str, Any]) -> dict[str, Any]:
    return predict_risk("diabetes", user_input)
