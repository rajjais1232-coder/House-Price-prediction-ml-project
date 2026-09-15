"""
analytics.py
------------
Business analytics, KPI calculations, and aggregations
for the Fraud Detection & Transaction Analytics Dashboard.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# KPI Calculation
# ---------------------------------------------------------------------------

def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate core business KPIs from the cleaned transaction DataFrame."""
    has_fraud = "Fraud" in df.columns

    total_transactions = len(df)
    total_amount = df["Amount"].sum() if "Amount" in df.columns else 0.0
    avg_amount = df["Amount"].mean() if "Amount" in df.columns else 0.0
    median_amount = df["Amount"].median() if "Amount" in df.columns else 0.0

    fraud_count = int(df["Fraud"].sum()) if has_fraud else None
    fraud_rate = float(df["Fraud"].mean() * 100) if has_fraud else None
    fraud_amount = float(df.loc[df["Fraud"] == 1, "Amount"].sum()) if has_fraud and "Amount" in df.columns else None
    fraud_amount_pct = float(fraud_amount / total_amount * 100) if fraud_amount and total_amount else None
    avg_fraud_amount = float(df.loc[df["Fraud"] == 1, "Amount"].mean()) if has_fraud and fraud_count else None

    unique_customers = df["Customer_ID"].nunique() if "Customer_ID" in df.columns else None
    unique_merchants = df["Merchant"].nunique() if "Merchant" in df.columns else None

    return {
        "Total Transactions":        total_transactions,
        "Total Amount":              round(total_amount, 2),
        "Avg Transaction Amount":    round(avg_amount, 2),
        "Median Transaction Amount": round(median_amount, 2),
        "Fraud Transactions":        fraud_count,
        "Fraud Rate (%)":            round(fraud_rate, 2) if fraud_rate is not None else None,
        "Fraud Amount":              round(fraud_amount, 2) if fraud_amount is not None else None,
        "Fraud Amount (%)":          round(fraud_amount_pct, 2) if fraud_amount_pct is not None else None,
        "Avg Fraud Amount":          round(avg_fraud_amount, 2) if avg_fraud_amount is not None else None,
        "Unique Customers":          unique_customers,
        "Unique Merchants":          unique_merchants,
    }


# ---------------------------------------------------------------------------
# Time Analysis
# ---------------------------------------------------------------------------

def transactions_by_day(df: pd.DataFrame) -> pd.DataFrame:
    col = "Transaction_Day"
    if col not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(col).agg(
        Transactions=("Amount", "count"),
        Total_Amount=("Amount", "sum"),
        Fraud_Count=("Fraud", "sum") if "Fraud" in df.columns else ("Amount", lambda x: 0),
    ).reset_index()
    grp.columns = [col, "Transactions", "Total_Amount", "Fraud_Count"]
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Transactions"] * 100).round(2)
    return grp.sort_values(col)


def transactions_by_month(df: pd.DataFrame) -> pd.DataFrame:
    col = "Transaction_Month"
    if col not in df.columns:
        return pd.DataFrame()
    has_fraud = "Fraud" in df.columns
    grp = df.groupby(col).agg(
        Transactions=("Amount", "count"),
        Total_Amount=("Amount", "sum"),
        Fraud_Count=("Fraud", "sum") if has_fraud else ("Amount", lambda x: 0),
    ).reset_index()
    grp.columns = [col, "Transactions", "Total_Amount", "Fraud_Count"]
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Transactions"] * 100).round(2)
    return grp.sort_values(col)


def fraud_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if "Transaction_Hour" not in df.columns or "Fraud" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Transaction_Hour").agg(
        Total=("Fraud", "count"),
        Fraud_Count=("Fraud", "sum"),
    ).reset_index()
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total"] * 100).round(2)
    return grp.sort_values("Transaction_Hour")


def fraud_trend_over_time(df: pd.DataFrame) -> pd.DataFrame:
    col = "Transaction_Day"
    if col not in df.columns or "Fraud" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(col).agg(
        Total=("Fraud", "count"),
        Fraud_Count=("Fraud", "sum"),
        Fraud_Amount=("Amount", lambda x: df.loc[x.index, "Amount"][df.loc[x.index, "Fraud"] == 1].sum()),
    ).reset_index()
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total"] * 100).round(2)
    return grp.sort_values(col)


# ---------------------------------------------------------------------------
# Customer Analysis
# ---------------------------------------------------------------------------

def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Customer_ID" not in df.columns:
        return pd.DataFrame()
    has_fraud = "Fraud" in df.columns
    agg = {
        "Amount": ["count", "sum", "mean"],
    }
    if has_fraud:
        agg["Fraud"] = ["sum"]
    if "Failed_Transactions" in df.columns:
        agg["Failed_Transactions"] = ["max"]

    grp = df.groupby("Customer_ID").agg(agg)
    grp.columns = ["_".join(c).strip() for c in grp.columns]
    grp = grp.rename(columns={
        "Amount_count": "Total_Transactions",
        "Amount_sum":   "Total_Amount",
        "Amount_mean":  "Avg_Amount",
        "Fraud_sum":    "Fraud_Count",
        "Failed_Transactions_max": "Max_Failed",
    }).reset_index()

    if has_fraud:
        grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total_Transactions"] * 100).round(2)

    return grp.sort_values("Fraud_Count", ascending=False) if has_fraud else grp


def high_risk_customers(df: pd.DataFrame, min_transactions: int = 3, fraud_rate_threshold: float = 30.0) -> pd.DataFrame:
    cust = customer_summary(df)
    if cust.empty or "Fraud_Rate" not in cust.columns:
        return pd.DataFrame()
    mask = (cust["Fraud_Rate"] >= fraud_rate_threshold) & (cust["Total_Transactions"] >= min_transactions)
    return cust[mask].sort_values("Fraud_Rate", ascending=False)


# ---------------------------------------------------------------------------
# Merchant Analysis
# ---------------------------------------------------------------------------

def merchant_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Merchant" not in df.columns:
        return pd.DataFrame()
    has_fraud = "Fraud" in df.columns
    agg = {"Amount": ["count", "sum", "mean"]}
    if has_fraud:
        agg["Fraud"] = ["sum"]
    if "Merchant_Category" in df.columns:
        df = df.copy()

    grp = df.groupby("Merchant").agg(agg)
    grp.columns = ["_".join(c) for c in grp.columns]
    grp = grp.rename(columns={
        "Amount_count": "Total_Transactions",
        "Amount_sum":   "Total_Amount",
        "Amount_mean":  "Avg_Amount",
        "Fraud_sum":    "Fraud_Count",
    }).reset_index()

    if "Merchant_Category" in df.columns:
        cat_map = df.groupby("Merchant")["Merchant_Category"].first()
        grp = grp.join(cat_map, on="Merchant")

    if has_fraud:
        grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total_Transactions"] * 100).round(2)
        grp["Fraud_Amount"] = df[df["Fraud"] == 1].groupby("Merchant")["Amount"].sum().reindex(grp["Merchant"]).values

    return grp.sort_values("Fraud_Count", ascending=False) if has_fraud else grp


def fraud_by_merchant_category(df: pd.DataFrame) -> pd.DataFrame:
    if "Merchant_Category" not in df.columns or "Fraud" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Merchant_Category").agg(
        Total=("Fraud", "count"),
        Fraud_Count=("Fraud", "sum"),
        Total_Amount=("Amount", "sum"),
        Fraud_Amount=("Amount", lambda x: x[df.loc[x.index, "Fraud"] == 1].sum()),
    ).reset_index()
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total"] * 100).round(2)
    return grp.sort_values("Fraud_Rate", ascending=False)


# ---------------------------------------------------------------------------
# Geographical Analysis
# ---------------------------------------------------------------------------

def fraud_by_location(df: pd.DataFrame) -> pd.DataFrame:
    if "Location" not in df.columns or "Fraud" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("Location").agg(
        Total=("Fraud", "count"),
        Fraud_Count=("Fraud", "sum"),
        Total_Amount=("Amount", "sum"),
        Fraud_Amount=("Amount", lambda x: x[df.loc[x.index, "Fraud"] == 1].sum()),
    ).reset_index()
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total"] * 100).round(2)
    return grp.sort_values("Fraud_Rate", ascending=False)


# ---------------------------------------------------------------------------
# Payment / Device / Transaction Type Analysis
# ---------------------------------------------------------------------------

def fraud_by_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Generic function to compute fraud rate grouped by any categorical column."""
    if col not in df.columns or "Fraud" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(col).agg(
        Total=("Fraud", "count"),
        Fraud_Count=("Fraud", "sum"),
        Total_Amount=("Amount", "sum"),
    ).reset_index()
    grp["Fraud_Rate"] = (grp["Fraud_Count"] / grp["Total"] * 100).round(2)
    return grp.sort_values("Fraud_Rate", ascending=False)


def international_fraud_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return fraud_by_column(df, "International_Transaction")
