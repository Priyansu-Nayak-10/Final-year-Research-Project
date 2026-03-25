from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

DEFAULT_DIABETES_FEATURE_ORDER = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_Male",
    "gender_Other",
    "smoking_history_current",
    "smoking_history_ever",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_not current",
]

_SMOKING_TO_COLUMN = {
    "current": "smoking_history_current",
    "ever": "smoking_history_ever",
    "former": "smoking_history_former",
    "never": "smoking_history_never",
    "not current": "smoking_history_not current",
    "not_current": "smoking_history_not current",
    "not-current": "smoking_history_not current",
    "notcurrent": "smoking_history_not current",
    "no info": "",
    "no_info": "",
    "no-info": "",
    "none": "",
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

    normalized = _normalize_text(value)
    return 1.0 if normalized in {"1", "true", "yes", "y"} else 0.0


def build_diabetes_features(
    user_input: Mapping[str, Any], feature_order: Iterable[str] | None = None
) -> pd.DataFrame:
    columns = (
        list(feature_order) if feature_order is not None else DEFAULT_DIABETES_FEATURE_ORDER
    )
    row = {column: 0.0 for column in columns}

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

    return pd.DataFrame([row], columns=columns)
