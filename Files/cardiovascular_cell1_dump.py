df = pd.read_csv("../data/cardiovascular.csv")
code
#VSC-6affd0f2
python
# Baseline model training (TRAIN baseline models and compare on validation)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42)
}

baseline_results = []
for name, model in models.items():
    model.fit(X_train_prep, y_train)
    preds = model.predict(X_val_prep)
    probs = model.predict_proba(X_val_prep)[:,1] if hasattr(model, 'predict_proba') else None
    baseline_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, preds),
        'Precision': precision_score(y_val, preds),
        'Recall': recall_score(y_val, preds),
        'F1': f1_score(y_val, preds),
        'ROC-AUC': roc_auc_score(y_val, probs) if probs is not None else np.nan,
    })

baseline_df = pd.DataFrame(baseline_results)[['Model','Accuracy','Precision','Recall','F1','ROC-AUC']]
display(baseline_df.sort_values('F1', ascending=False).reset_index(drop=True))

for name, model in models.items():
    preds = model.predict(X_val_prep)
    cm = confusion_matrix(y_val, preds)
    print(f\
)
    display(cm)
code
python
# Optuna optimization utilities
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

code
python
# Optuna: Logistic Regression (20 trials)
def objective_lr(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    solver = trial.suggest_categorical('solver', ['liblinear','saga','lbfgs'])
    model = LogisticRegression(C=C, solver=solver, max_iter=1000)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, scoring=scorer, n_jobs=-1)
    return float(scores.mean())

study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=20)
print('LR best params:', study_lr.best_params)
optimized_lr = LogisticRegression(**study_lr.best_params, max_iter=1000)
optimized_lr.fit(X_train_prep, y_train)
code
python
# Optuna: Random Forest (20 trials)
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, scoring=scorer, n_jobs=-1)
    return float(scores.mean())

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=20)
print('RF best params:', study_rf.best_params)
optimized_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
optimized_rf.fit(X_train_prep, y_train)
code
python
# Optuna: XGBoost (30 trials)
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 12)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1.0)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                          subsample=subsample, colsample_bytree=colsample_bytree,
                          use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, scoring=scorer, n_jobs=-1)
    return float(scores.mean())

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
print('XGB best params:', study_xgb.best_params)
optimized_xgb = XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
optimized_xgb.fit(X_train_prep, y_train)
code
python
# Optuna: LightGBM (30 trials)
from lightgbm import LGBMClassifier
def objective_lgb(trial):
    num_leaves = trial.suggest_int('num_leaves', 8, 256)
    max_depth = trial.suggest_int('max_depth', -1, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    model = LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                           n_estimators=n_estimators, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_train_prep, y_train, cv=cv, scoring=scorer, n_jobs=-1)
    return float(scores.mean())

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
print('LGBM best params:', study_lgb.best_params)
optimized_lgbm = LGBMClassifier(**study_lgb.best_params, random_state=42, n_jobs=-1)
optimized_lgbm.fit(X_train_prep, y_train)
code
python
# Optimized models comparison on TEST set
optimized_models = {
    'Logistic Regression': optimized_lr,
    'Random Forest': optimized_rf,
    'XGBoost': optimized_xgb,
    'LightGBM': optimized_lgbm
}

def eval_model_on_test(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_proba) if y_proba is not None else np.nan
    }

rows = []
for name, model in optimized_models.items():
    metrics = eval_model_on_test(model, X_test_prep, y_test)
    metrics['Model'] = name
    rows.append(metrics)

optimized_results_df = pd.DataFrame(rows)[['Model','Accuracy','Precision','Recall','F1','ROC-AUC']]
display(optimized_results_df.sort_values('F1', ascending=False).reset_index(drop=True))
code
python
# Select best optimized model by highest F1
best_row = optimized_results_df.sort_values('F1', ascending=False).iloc[0]
best_model_name = best_row['Model']
best_model = optimized_models[best_model_name]
print(f\
code
python
# Threshold tuning for best_model using validation set
val_X = X_val_prep
val_y = y_val
probs = best_model.predict_proba(val_X)[:,1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(val_X)
thresholds = np.linspace(0.01, 0.99, 99)
best_thr = 0.5
best_f1 = 0.0
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    f1 = f1_score(val_y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print('Optimal threshold:', best_thr, 'F1:', best_f1)
threshold = float(best_thr)
code
python
# Final evaluation using the tuned threshold on TEST set
probs_test = best_model.predict_proba(X_test_prep)[:,1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test_prep)
y_pred_adj = (probs_test >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_adj)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_adj),
    'Precision': precision_score(y_test, y_pred_adj),
    'Recall': recall_score(y_test, y_pred_adj),
    'F1': f1_score(y_test, y_pred_adj),
    'ROC-AUC': roc_auc_score(y_test, probs_test) if probs_test is not None else np.nan
}
print('Final metrics (thresholded):', metrics)
print('Confusion matrix:')
print(cm)
code
python
# Save artifacts for Streamlit-compatible inference
out_dir = os.path.join('..','models')
os.makedirs(out_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(out_dir, 'cardio_best_model.joblib'))
if 'preprocessor' in globals():
    joblib.dump(preprocessor, os.path.join(out_dir, 'cardio_preprocessor.joblib'))
if 'selector' in globals():
    joblib.dump(selector, os.path.join(out_dir, 'cardio_selector.joblib'))
joblib.dump(selected_features, os.path.join(out_dir, 'cardio_selected_features.joblib'))
joblib.dump(threshold, os.path.join(out_dir, 'cardio_threshold.joblib'))
joblib.dump(list(X.columns), os.path.join(out_dir, 'cardio_feature_columns.joblib'))
with open(os.path.join(out_dir, 'cardio_best_model_name.txt'), 'w') as f:
    f.write(best_model_name)
print('Artifacts saved to', out_dir)
markdown
markdown
# Comments
SHAP removed from this notebook per user instruction; no SHAP cells remain.
code
#VSC-3408e87a
python
# Validation-set evaluation
results = []
 
def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall":    recall_score(y, y_pred),
        "F1 Score":  f1_score(y, y_pred),
    })
 
for name, model in models.items():
    evaluate_model(name, model, X_val_prep, y_val)
 
results_df = pd.DataFrame(results)
print("\n-- Validation Comparison --")
print(results_df.round(4).to_string(index=False))
 
# Bar chart
results_df.set_index("Model").plot(kind="bar", figsize=(10, 5))
plt.title("Validation Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()
code
#VSC-5cf030d3
python
# Threshold tuning - find best decision threshold on validation set only
print("\n-- Threshold Tuning --")
best_thresholds  = {}
best_f1_scores   = {}
thresholds_range = [round(i * 0.05, 2) for i in range(6, 15)]   # 0.30 -> 0.70
 
for name, model in models.items():
    probs   = model.predict_proba(X_val_prep)[:, 1]
    best_f1 = 0.0
    best_t  = 0.5
 
    for t in thresholds_range:
        score = f1_score(y_val, (probs >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_t  = t
 
    best_thresholds[name] = best_t
    best_f1_scores[name]  = best_f1
    print(f"{name:22s}: best threshold = {best_t}, F1 = {best_f1:.4f}")
code
#VSC-c21eaaee
python
# Select best model using VALIDATION performance (not test)
best_model_name = max(best_f1_scores, key=best_f1_scores.get)
best_model      = models[best_model_name]
best_threshold  = best_thresholds[best_model_name]
best_f1_val     = best_f1_scores[best_model_name]
 
print(f"\n Best Model : {best_model_name}")
print(f"   Threshold   : {best_threshold}")
print(f"   Val F1      : {best_f1_val:.4f}")

code
#VSC-d7b792ab
python
# Reusable probability-to-severity mapping

def get_risk_level(probability):
    prob_percent = probability * 100
    if prob_percent <= 30:
        return "Low Risk"
    if prob_percent <= 60:
        return "Moderate Risk"
    if prob_percent <= 80:
        return "High Risk"
    return "Critical Risk"


# Test set evaluation with best model and optimized threshold
y_test_prob = best_model.predict_proba(X_test_prep)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\n-- Final Test Performance  [{best_model_name}  |  threshold={best_threshold}] --")
print(f"Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_test_pred):.4f}")
print(f"Sample Risk Severity (first test sample): {get_risk_level(y_test_prob[0])}")

# Confusion matrix chart
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

code
#VSC-a832453e
python
# SHAP explainability for the optimized XGBoost model
X_train_selected_df = pd.DataFrame(X_train_prep, columns=selected_features)

shap_explainer = shap.TreeExplainer(best_model)
shap_values = shap_explainer.shap_values(X_train_selected_df)

# For binary classifiers, SHAP can return a list; use positive-class contribution
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("\n-- SHAP Summary Plot --")
shap.summary_plot(shap_values, X_train_selected_df, feature_names=selected_features)

print("\n-- SHAP Feature Importance (Bar) --")
shap.summary_plot(
    shap_values,
    X_train_selected_df,
    feature_names=selected_features,
    plot_type="bar"
)

code
#VSC-8b933ba2
python
# Save complete artifact with all components needed for inference
# SHAP compatibility is preserved via model + selected feature mapping.
final_artifact = {
    "model": best_model,
    "preprocessor": preprocessor,
    "selector": selector,
    "threshold": best_threshold,
    "selected_features": selected_features,
    "feature_columns": list(X.columns),
    "features": list(X.columns),
    "model_name": best_model_name,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
}

joblib.dump(final_artifact, "../models/cardiovascular_model.pkl")

print("\nHeart model saved successfully!")
print(f"Model: {best_model_name}")
print(f"Threshold: {best_threshold}")
print(f"Selector: SelectKBest(k={best_k})")
print("Preprocessor: StandardScaler + OneHotEncoder")

code
#VSC-cf0bb3b3
python
artifact = joblib.load("../models/cardiovascular_model.pkl")

model = artifact["model"]
preprocessor = artifact["preprocessor"]
selector = artifact["selector"]
threshold = artifact["threshold"]
numeric_cols = artifact["numeric_cols"]
categorical_cols = artifact["categorical_cols"]
features = artifact["features"]
selected_features = artifact["selected_features"]

# Keep risk helper reusable even when running this cell independently.
def get_risk_level(probability):
    prob_percent = probability * 100
    if prob_percent <= 30:
        return "Low Risk"
    if prob_percent <= 60:
        return "Moderate Risk"
    if prob_percent <= 80:
        return "High Risk"
    return "Critical Risk"

data = {
    "age": 72,
    "bmi": 31.8,
    "systolic_bp": 161,
    "diastolic_bp": 111,
    "cholesterol_mg_dl": 261,
    "resting_heart_rate": 73,
    "smoking_status": "Current",
    "daily_steps": 3435,
    "stress_level": 2,
    "physical_activity_hours_per_week": 3.3,
    "sleep_hours": 7.9,
    "family_history_heart_disease": "No",
    "diet_quality_score": 2,
    "alcohol_units_per_week": 0.8
}

row = pd.DataFrame([data])

# Apply the same preprocessing stack used during training
X_new = preprocessor.transform(row)
X_new = selector.transform(X_new)

prob = model.predict_proba(X_new)[0, 1]
pred = 1 if prob >= threshold else 0

print("Risk:", "High" if pred == 1 else "Low")
print("Risk Level:", get_risk_level(prob))
print("Probability:", round(prob, 3))
print("Threshold:", threshold)
