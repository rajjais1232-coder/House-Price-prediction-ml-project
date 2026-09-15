"""
feature_engineering.py
-----------------------
Prepares features and target for ML model training and inference.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Dict, Optional
import joblib
import os

TARGET = "Fraud"

# Columns used as ML features (canonical names)
FEATURE_CANDIDATES = [
    "Amount",
    "Customer_Age",
    "Account_Age",
    "Previous_Transactions",
    "Failed_Transactions",
    "Transaction_Frequency",
    "Merchant_Category",
    "Location",
    "Transaction_Type",
    "Payment_Method",
    "Device_Type",
    "International_Transaction",
]

# Columns to scale
SCALE_COLS = [
    "Amount",
    "Customer_Age",
    "Account_Age",
    "Previous_Transactions",
    "Failed_Transactions",
    "Transaction_Frequency",
]

# Columns to label-encode
ENCODE_COLS = [
    "Merchant_Category",
    "Location",
    "Transaction_Type",
    "Payment_Method",
    "Device_Type",
    "International_Transaction",
]


def get_available_features(df: pd.DataFrame) -> List[str]:
    """Return the subset of FEATURE_CANDIDATES that actually exist in df."""
    return [c for c in FEATURE_CANDIDATES if c in df.columns]


def build_feature_matrix(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder], StandardScaler]:
    """
    Encode and scale features.

    Parameters
    ----------
    df      : cleaned DataFrame (canonical column names)
    encoders: pre-fitted encoders (for inference); ignored when fit=True
    scaler  : pre-fitted scaler  (for inference); ignored when fit=True
    fit     : if True, fit new encoders/scaler; if False, use provided ones

    Returns
    -------
    X          : feature DataFrame
    feature_names : list of feature column names
    encoders   : fitted LabelEncoders dict
    scaler     : fitted StandardScaler
    """
    features = get_available_features(df)
    X = df[features].copy()

    # ── Encode categoricals ─────────────────────────────────────────────────
    if encoders is None:
        encoders = {}

    encode_cols = [c for c in ENCODE_COLS if c in X.columns]
    for col in encode_cols:
        X[col] = X[col].astype(str).fillna("Unknown")
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])
            else:
                X[col] = 0  # fallback if encoder missing

    # ── Scale numerics ───────────────────────────────────────────────────────
    scale_cols = [c for c in SCALE_COLS if c in X.columns]
    for col in scale_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    if scaler is None:
        scaler = StandardScaler()

    if scale_cols:
        if fit:
            X[scale_cols] = scaler.fit_transform(X[scale_cols])
        else:
            X[scale_cols] = scaler.transform(X[scale_cols])

    feature_names = list(X.columns)
    return X, feature_names, encoders, scaler


def prepare_single_transaction(
    txn: Dict,
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Prepare a single transaction dict for inference.
    Returns a 1-row DataFrame with the correct feature matrix.
    """
    row = pd.DataFrame([txn])
    # Fill missing features with 0
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    # Encode
    for col in ENCODE_COLS:
        if col in row.columns:
            row[col] = row[col].astype(str)
            le = encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                row[col] = row[col].apply(lambda v: v if v in known else le.classes_[0])
                row[col] = le.transform(row[col])
            else:
                row[col] = 0

    # Scale
    scale_cols = [c for c in SCALE_COLS if c in row.columns]
    for col in scale_cols:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)
    if scale_cols:
        row[scale_cols] = scaler.transform(row[scale_cols])

    return row[feature_names]


def save_artifacts(
    model,
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    feature_names: List[str],
    model_name: str = "fraud_model",
    save_dir: str = "models",
) -> str:
    """Save model + preprocessing artefacts to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(
        {
            "model":         model,
            "encoders":      encoders,
            "scaler":        scaler,
            "feature_names": feature_names,
        },
        path,
    )
    return path


def load_artifacts(path: str) -> Dict:
    """Load model + preprocessing artefacts from disk."""
    return joblib.load(path)
