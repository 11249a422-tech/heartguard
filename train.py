"""
HeartGuard ML Training Pipeline
Loads UCI Cleveland dataset, preprocesses features, benchmarks 3 models
(Logistic Regression, Random Forest, XGBoost), selects the best by ROC-AUC,
and serializes the pipeline and evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_and_preprocess_data(data_path=DATA_PATH):
    """Load data and split into features and binary target."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path, na_values="?")

    # Features and binary target (0 = no disease, 1 = disease present)
    X = df.drop(columns=["target"])
    y = (df["target"] > 0).astype(int)

    return X, y


def build_preprocessor(numeric_features, categorical_features):
    """Build ColumnTransformer for numeric and categorical pipelines."""
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train_and_evaluate():
    """Train models, evaluate on test set, save champion model and metrics."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_and_preprocess_data()

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = {}
    fitted_pipelines = {}

    print("=" * 60)
    print("HeartGuard ML Model Training & Benchmarking")
    print("=" * 60)
    print(f"Dataset Size: {len(X)} rows ({len(X_train)} train, {len(X_test)} test)")
    print(f"Positive cases: {y.sum()} ({y.mean()*100:.1f}%), Negative cases: {(y == 0).sum()} ({(1-y.mean())*100:.1f}%)\n")

    for name, clf in candidate_models.items():
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }
        fitted_pipelines[name] = pipe

        print(f"[{name}]")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1 Score  : {f1*100:.2f}%")
        print(f"  ROC-AUC   : {auc*100:.2f}%\n")

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipe = fitted_pipelines[best_name]

    print("=" * 60)
    print(f"WINNER: {best_name} (ROC-AUC: {results[best_name]['roc_auc']*100:.2f}%)")
    print("=" * 60)

    # Save champion model
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Successfully saved champion model pipeline to {MODEL_PATH}")

    # Extract feature names & importances/coefficients for interpretability
    feature_importances = {}
    preprocessor = best_pipe.named_steps["preprocessor"]
    classifier = best_pipe.named_steps["classifier"]

    # Retrieve transformed feature names
    try:
        encoded_cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(encoded_cat_names)
    except Exception:
        all_features = numeric_features + categorical_features

    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
        feature_importances = {feat: round(float(imp), 4) for feat, imp in zip(all_features, raw_importances)}
    elif hasattr(classifier, "coef_"):
        raw_coefs = classifier.coef_[0]
        feature_importances = {feat: round(float(coef), 4) for feat, coef in zip(all_features, raw_coefs)}

    # Top overall features
    top_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    metrics_payload = {
        "dataset": {
            "name": "UCI Cleveland Heart Disease",
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_count": int(X.shape[1]),
        },
        "models": results,
        "best_model": best_name,
        "best_metrics": results[best_name],
        "top_features": dict(top_features),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to {METRICS_PATH}")

    return best_pipe, metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
