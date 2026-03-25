from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes.csv"
OUTPUT_DIR = ROOT_DIR / "models" / "diabetes"


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, drop_first=True).dropna()

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_baseline = StandardScaler()
    X_train_scaled = scaler_baseline.fit_transform(X_train)
    X_test_scaled = scaler_baseline.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    baseline_rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        if name in {"Logistic Regression", "SVM"}:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        baseline_rows.append({"Model": name, **evaluate_binary(y_test, y_pred)})

    baseline_df = pd.DataFrame(baseline_rows).sort_values(by="F1", ascending=False)
    top_models = baseline_df.head(2)["Model"].tolist()

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled_smote = scaler.transform(X_test)

    tuned_results: dict[str, dict] = {}
    for name in top_models:
        model = models[name]

        if name in {"Logistic Regression", "SVM"}:
            model.fit(X_train_res_scaled, y_train_res)
            probs = model.predict_proba(X_test_scaled_smote)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            probs = model.predict_proba(X_test)[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= threshold).astype(int)
            score = f1_score(y_test, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        final_preds = (probs >= best_threshold).astype(int)
        tuned_results[name] = {
            "model": model,
            "Threshold": best_threshold,
            **evaluate_binary(y_test, final_preds),
        }

    best_model_name = max(tuned_results, key=lambda n: tuned_results[n]["F1"])
    best_model = tuned_results[best_model_name]["model"]
    best_threshold = tuned_results[best_model_name]["Threshold"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUTPUT_DIR / "model.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(best_threshold, OUTPUT_DIR / "threshold.pkl")

    print("Best model:", best_model_name)
    print("Saved model artifacts to:", OUTPUT_DIR)
    print("Chosen threshold:", best_threshold)


if __name__ == "__main__":
    main()
