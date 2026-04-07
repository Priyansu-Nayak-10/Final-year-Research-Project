import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "Lung Cancer.csv"
MODEL_PATH = ROOT_DIR / "models" / "lung_model.pkl"
TARGET_COL = "pulmonary_disease"
POSITIVE_LABEL = "YES"
NEGATIVE_LABEL = "NO"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def evaluate_with_threshold(model, x, y, threshold: float):
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }


def main():
    df = pd.read_csv(DATASET_PATH)
    df = preprocess_dataframe(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[TARGET_COL].astype(str).str.upper().map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), TARGET_COL].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target labels found: {invalid}")

    x = df.drop(columns=[TARGET_COL]).copy()
    feature_columns = x.columns.tolist()

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_val_imp = imputer.transform(x_val)
    x_test_imp = imputer.transform(x_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_val_scaled = scaler.transform(x_val_imp)
    x_test_scaled = scaler.transform(x_test_imp)

    selector = SelectKBest(score_func=f_classif, k=min(12, x_train_scaled.shape[1]))
    x_train_sel = selector.fit_transform(x_train_scaled, y_train)
    x_val_sel = selector.transform(x_val_scaled)
    x_test_sel = selector.transform(x_test_scaled)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(x_train_sel, y_train)
        val_preds = model.predict(x_val_sel)
        model_scores[name] = float(f1_score(y_val, val_preds, zero_division=0))

    best_model_name = max(model_scores, key=model_scores.get)
    final_model = models[best_model_name]

    val_probs = final_model.predict_proba(x_val_sel)[:, 1]
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.2, 0.901, 0.01):
        val_preds = (val_probs >= threshold).astype(int)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(round(threshold, 2))

    val_metrics = evaluate_with_threshold(final_model, x_val_sel, y_val, best_threshold)
    test_metrics = evaluate_with_threshold(final_model, x_test_sel, y_test, best_threshold)

    selected_features = [
        feature for feature, keep in zip(feature_columns, selector.get_support()) if keep
    ]

    model_data = {
        "model": final_model,
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "threshold": best_threshold,
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        "target_column": TARGET_COL,
        "positive_label": "Lung Cancer Risk (YES)",
        "negative_label": "Lung Cancer Risk (NO)",
        "target_definition": "pulmonary_disease == 'YES'",
        "source_dataset": str(DATASET_PATH),
        "train_note": "Model/threshold selected on validation split; test kept unseen.",
        "model_candidates_f1_val": model_scores,
        "best_model_name": best_model_name,
        "best_threshold_f1_val": best_f1,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_PATH)

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_cols": int(df.shape[1]),
        "class_balance": {
            "0_no": int((y == 0).sum()),
            "1_yes": int((y == 1).sum()),
        },
        "saved_model_path": str(MODEL_PATH),
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "selected_features_count": len(selected_features),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
