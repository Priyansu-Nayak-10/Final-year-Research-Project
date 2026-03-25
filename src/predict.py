from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.feature_builder import build_diabetes_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "diabetes_threshold.pkl"
SCALER_PATH = MODEL_DIR / "diabetes_scaler.pkl"


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


def predict_diabetes(user_input: Mapping[str, Any]) -> dict[str, Any]:
    model, threshold, scaler = load_diabetes_artifacts()
    features_df = build_diabetes_features(user_input)

    model_input = features_df
    if scaler is not None:
        scaled = scaler.transform(features_df)
        model_input = pd.DataFrame(scaled, columns=features_df.columns)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])
    else:
        probability = float(model.predict(model_input)[0])

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "High diabetes risk" if prediction == 1 else "Lower diabetes risk",
        "probability": probability,
        "threshold": threshold,
        "model_input": features_df.iloc[0].to_dict(),
    }
