from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.utils.helpers import DEFAULT_DIABETES_FEATURE_ORDER, build_diabetes_features

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "diabetes"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
_SCALED_MODEL_CLASS_NAMES = {"LogisticRegression", "SVC"}


@lru_cache(maxsize=1)
def load_diabetes_artifacts() -> tuple[Any, float, Any | None]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, threshold, scaler


def _infer_feature_order(model: Any, scaler: Any | None) -> list[str]:
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(DEFAULT_DIABETES_FEATURE_ORDER)


def _requires_scaling(model: Any, scaler: Any | None) -> bool:
    if scaler is None:
        return False
    return model.__class__.__name__ in _SCALED_MODEL_CLASS_NAMES


def predict_diabetes(user_input: Mapping[str, Any]) -> dict[str, Any]:
    model, threshold, scaler = load_diabetes_artifacts()
    feature_order = _infer_feature_order(model, scaler)
    features_df = build_diabetes_features(user_input, feature_order=feature_order)

    model_input = features_df
    uses_scaler = _requires_scaling(model, scaler)
    if uses_scaler:
        scaled = scaler.transform(features_df)
        model_input = pd.DataFrame(scaled, columns=features_df.columns)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(model_input)[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))
    else:
        probability = float(model.predict(model_input)[0])

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "High diabetes risk" if prediction == 1 else "Lower diabetes risk",
        "probability": probability,
        "threshold": threshold,
        "model_input": features_df.iloc[0].to_dict(),
        "model_name": model.__class__.__name__,
        "uses_scaler": uses_scaler,
    }
