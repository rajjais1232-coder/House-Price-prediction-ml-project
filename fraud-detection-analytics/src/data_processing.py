"""
data_processing.py
------------------
Handles CSV ingestion, column mapping, cleaning, and preprocessing
for the Fraud Detection & Transaction Analytics Dashboard.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# ---------------------------------------------------------------------------
# Expected column schema with aliases
# ---------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Transaction_ID":          ["transaction_id", "txn_id", "id", "trans_id"],
    "Customer_ID":             ["customer_id", "cust_id", "user_id", "client_id"],
    "Transaction_Date":        ["transaction_date", "date", "trans_date", "txn_date"],
    "Transaction_Time":        ["transaction_time", "time", "txn_time", "trans_time"],
    "Amount":                  ["amount", "transaction_amount", "amt", "value"],
    "Merchant":                ["merchant", "merchant_name", "vendor", "store"],
    "Merchant_Category":       ["merchant_category", "category", "mcc", "merchant_type"],
    "Location":                ["location", "city", "state", "region", "address"],
    "Transaction_Type":        ["transaction_type", "type", "txn_type", "trans_type"],
    "Payment_Method":          ["payment_method", "payment_type", "pay_method", "method"],
    "Device_Type":             ["device_type", "device", "platform", "channel"],
    "Customer_Age":            ["customer_age", "age", "cust_age"],
    "Account_Age":             ["account_age", "acct_age", "account_tenure"],
    "Previous_Transactions":   ["previous_transactions", "prev_transactions", "history_count"],
    "Failed_Transactions":     ["failed_transactions", "failed_txns", "declined"],
    "Transaction_Frequency":   ["transaction_frequency", "freq", "frequency"],
    "International_Transaction":["international_transaction", "international", "is_international"],
    "Fraud":                   ["fraud", "is_fraud", "label", "fraud_flag", "target"],
}

NUMERIC_COLS = [
    "Amount", "Customer_Age", "Account_Age",
    "Previous_Transactions", "Failed_Transactions", "Transaction_Frequency",
]
CATEGORICAL_COLS = [
    "Merchant", "Merchant_Category", "Location",
    "Transaction_Type", "Payment_Method", "Device_Type",
    "International_Transaction",
]
DATE_COL = "Transaction_Date"
TIME_COL = "Transaction_Time"
TARGET_COL = "Fraud"


# ---------------------------------------------------------------------------
# Column detection & mapping
# ---------------------------------------------------------------------------

def detect_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Return a mapping {canonical_name: actual_column_name | None}.
    Performs case-insensitive, underscore-normalised matching.
    """
    df_cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
    mapping: Dict[str, Optional[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = [canonical.lower()] + aliases
        found = None
        for alias in candidates:
            if alias in df_cols_lower:
                found = df_cols_lower[alias]
                break
        mapping[canonical] = found
    return mapping


def apply_column_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Rename columns according to the mapping (skip unmapped columns).
    Returns a new DataFrame with canonical column names.
    """
    rename_map = {v: k for k, v in mapping.items() if v is not None and v != k}
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Dataset quality report
# ---------------------------------------------------------------------------

def dataset_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a dictionary describing data quality issues.
    Does NOT modify df.
    """
    total = len(df)
    report: Dict[str, Any] = {
        "total_rows": total,
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "missing_pct_by_column": (df.isnull().sum() / total * 100).round(2).to_dict(),
        "total_missing_cells": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    # Numeric outlier counts (IQR method)
    outlier_counts: Dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        outlier_counts[col] = n_outliers
    report["outlier_counts"] = outlier_counts

    # Class balance (if Fraud column present)
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts()
        report["class_balance"] = vc.to_dict()
        report["fraud_rate"] = round(float((df[TARGET_COL] == 1).mean() * 100), 2)
    else:
        report["class_balance"] = None
        report["fraud_rate"] = None

    return report


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a cleaned copy of df and a list of cleaning actions applied.
    Original df is NOT modified.
    """
    df = df.copy()
    actions: List[str] = []

    # 1. Remove exact duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df.drop_duplicates(inplace=True)
        actions.append(f"Removed {n_dupes} duplicate rows.")

    # 2. Parse date column
    if DATE_COL in df.columns:
        try:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
            n_bad_dates = df[DATE_COL].isna().sum()
            if n_bad_dates:
                actions.append(f"Coerced {n_bad_dates} unparseable dates to NaT.")
        except Exception:
            actions.append(f"Could not parse {DATE_COL} column.")

    # 3. Parse Amount column
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        n_bad = df["Amount"].isna().sum()
        if n_bad:
            actions.append(f"Coerced {n_bad} non-numeric Amount values to NaN.")
        # Remove negative amounts
        n_neg = (df["Amount"] < 0).sum()
        if n_neg:
            df = df[df["Amount"] >= 0]
            actions.append(f"Removed {n_neg} rows with negative Amount.")

    # 4. Parse Fraud column
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df[TARGET_COL] = df[TARGET_COL].map(
            lambda x: 1 if x == 1 or str(x).lower() in ("yes", "true", "1") else 0
        )

    # 5. Fill numeric missing values with median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().sum():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            actions.append(f"Filled missing {col} with median ({median_val:.2f}).")

    # 6. Fill categorical missing values with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().sum():
            mode_val = df[col].mode()
            if len(mode_val):
                df[col].fillna(mode_val[0], inplace=True)
                actions.append(f"Filled missing {col} with mode ('{mode_val[0]}').")

    # 7. Standardise International_Transaction to Yes/No
    if "International_Transaction" in df.columns:
        intl = df["International_Transaction"].astype(str).str.strip().str.lower()
        df["International_Transaction"] = intl.map(
            lambda x: "Yes" if x in ("yes", "1", "true", "y") else "No"
        )

    # 8. Derive Time Features
    if DATE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df["Transaction_Month"]      = df[DATE_COL].dt.to_period("M").astype(str)
        df["Transaction_DayOfWeek"]  = df[DATE_COL].dt.day_name()
        df["Transaction_Day"]        = df[DATE_COL].dt.date.astype(str)

    if TIME_COL in df.columns:
        try:
            t = pd.to_datetime(df[TIME_COL], format="%H:%M:%S", errors="coerce")
            df["Transaction_Hour"] = t.dt.hour
        except Exception:
            pass

    actions.append(f"Cleaned dataset: {len(df)} rows, {len(df.columns)} columns.")
    return df, actions


# ---------------------------------------------------------------------------
# Feature summary for display
# ---------------------------------------------------------------------------

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising each column."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique())
        sample = df[col].dropna().iloc[:3].tolist() if len(df[col].dropna()) else []
        rows.append({
            "Column": col,
            "Dtype": dtype,
            "Missing": n_missing,
            "Missing %": round(n_missing / len(df) * 100, 1),
            "Unique Values": n_unique,
            "Sample Values": str(sample),
        })
    return pd.DataFrame(rows)
