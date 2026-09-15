"""
model_evaluation.py
-------------------
Evaluation utilities: metrics, confusion matrix, feature importance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from typing import Any, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Return a dict of evaluation metrics for one model."""
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X_test)

    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    avg_prec   = average_precision_score(y_test, y_prob) if y_prob is not None else None

    return {
        "Model":     model_name,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1 Score":  round(f1, 4),
        "ROC-AUC":   round(roc_auc, 4) if roc_auc is not None else None,
        "Avg Precision": round(avg_prec, 4) if avg_prec is not None else None,
    }


def get_confusion_matrix_fig(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> go.Figure:
    """Return a Plotly heatmap of the confusion matrix."""
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate", "Fraudulent"]

    fig = px.imshow(
        cm,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(height=400)
    return fig


def get_feature_importance_fig(
    model,
    feature_names: list,
    model_name: str = "Model",
    top_n: int = 15,
) -> Optional[go.Figure]:
    """Return a horizontal bar chart of feature importances for tree models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=True).tail(top_n)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Feature Importance — {model_name}",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=450, showlegend=False)
    return fig


def get_roc_pr_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[go.Figure, go.Figure]:
    """
    Return ROC curve and Precision-Recall curve figures for all models.
    """
    from sklearn.metrics import roc_curve

    roc_fig = go.Figure()
    pr_fig  = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, (name, model) in enumerate(models_dict.items()):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        color = colors[i % len(colors)]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={auc_val:.3f})",
            mode="lines",
            line=dict(color=color),
        ))

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        avg_p = average_precision_score(y_test, y_prob)
        pr_fig.add_trace(go.Scatter(
            x=rec, y=prec,
            name=f"{name} (AP={avg_p:.3f})",
            mode="lines",
            line=dict(color=color),
        ))

    # Diagonal reference for ROC
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random",
        line=dict(dash="dash", color="gray"),
    ))

    roc_fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    pr_fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
    )

    return roc_fig, pr_fig


# ---------------------------------------------------------------------------
# Fraud probability risk categorisation
# ---------------------------------------------------------------------------

def categorise_risk(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def risk_color(risk_label: str) -> str:
    if "Low" in risk_label:
        return "green"
    elif "Medium" in risk_label:
        return "orange"
    else:
        return "red"
