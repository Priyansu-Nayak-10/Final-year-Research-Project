from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

EXPECTED_FEATURE_ORDER = [
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


def build_diabetes_features(user_input: Mapping[str, Any]) -> pd.DataFrame:
    row = {column: 0.0 for column in EXPECTED_FEATURE_ORDER}

    row["age"] = _to_float(user_input.get("age"))
    row["hypertension"] = _to_binary(user_input.get("hypertension"))
    row["heart_disease"] = _to_binary(user_input.get("heart_disease"))
    row["bmi"] = _to_float(user_input.get("bmi"))
    row["HbA1c_level"] = _to_float(user_input.get("HbA1c_level"))
    row["blood_glucose_level"] = _to_float(user_input.get("blood_glucose_level"))

    gender = _normalize_text(user_input.get("gender", "female"))
    if gender in {"male", "m"}:
        row["gender_Male"] = 1.0
    elif gender in {"other", "o"}:
        row["gender_Other"] = 1.0

    smoking_history = _normalize_text(user_input.get("smoking_history", "not current"))
    smoking_column = _SMOKING_TO_COLUMN.get(smoking_history)
    if smoking_column:
        row[smoking_column] = 1.0

    return pd.DataFrame([row], columns=EXPECTED_FEATURE_ORDER)
