from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "heart"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"


def predict_heart(_user_input: dict) -> dict:
    raise NotImplementedError(
        "Heart prediction is not implemented yet. Add heart preprocessing, training, and artifacts first."
    )
