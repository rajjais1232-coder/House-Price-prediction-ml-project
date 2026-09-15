"""
model_training.py
-----------------
Trains and compares multiple fraud-detection models.
Handles class imbalance using class_weight='balanced'.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from src.feature_engineering import build_feature_matrix, save_artifacts, TARGET

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODELS_DIR = "models"
ARTIFACT_PATH = "models/fraud_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _get_model_definitions() -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=9,  # approximate imbalance correction
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    return models


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_all_models(
    df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict, Any, list]:
    """
    Train all models and return results.

    Returns
    -------
    trained_models : dict {name: fitted model}
    comparison_df  : DataFrame with metrics per model
    best_artifacts : {model, encoders, scaler, feature_names}
    best_model_name: str
    feature_names  : list of feature column names
    """
    from src.model_evaluation import evaluate_model

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    # Prepare features
    X, feature_names, encoders, scaler = build_feature_matrix(df, fit=True)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model_defs = _get_model_definitions()
    trained_models: Dict[str, Any] = {}
    metrics_rows = []

    for name, model in model_defs.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            metrics = evaluate_model(model, X_test, y_test, name)
            metrics_rows.append(metrics)
        except Exception as e:
            metrics_rows.append({"Model": name, "Error": str(e)})

    comparison_df = pd.DataFrame(metrics_rows)

    # Select best model by F1 Score
    if "F1 Score" in comparison_df.columns:
        best_idx = comparison_df["F1 Score"].fillna(0).idxmax()
        best_name = comparison_df.loc[best_idx, "Model"]
    else:
        best_name = list(trained_models.keys())[-1]

    best_model = trained_models[best_name]
    artifact_path = save_artifacts(
        model=best_model,
        encoders=encoders,
        scaler=scaler,
        feature_names=feature_names,
        model_name="fraud_model",
        save_dir=MODELS_DIR,
    )

    best_artifacts = {
        "model":         best_model,
        "encoders":      encoders,
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    best_name,
        "artifact_path": artifact_path,
        "X_test":        X_test,
        "y_test":        y_test,
    }

    return trained_models, comparison_df, best_artifacts, best_name, feature_names


# ---------------------------------------------------------------------------
# Anomaly Detection — Isolation Forest
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection on numerical features.
    Returns df with an 'Anomaly' column (-1 = anomaly, 1 = normal)
    and an 'Anomaly_Score' column.
    """
    num_cols = [
        c for c in ["Amount", "Failed_Transactions", "Account_Age",
                     "Previous_Transactions", "Transaction_Frequency"]
        if c in df.columns
    ]
    if len(num_cols) < 2:
        raise ValueError("Not enough numeric features for anomaly detection.")

    X = df[num_cols].copy()
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    df = df.copy()
    df["Anomaly"] = iso.fit_predict(X)
    df["Anomaly_Score"] = iso.score_samples(X)  # lower = more anomalous
    return df
