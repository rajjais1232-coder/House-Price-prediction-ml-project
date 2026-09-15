# -*- coding: utf-8 -*-
"""
app.py
------
Streamlit frontend for the House Price Prediction project.

Tabs:
  - Predict  : form inputs → predicted median house value
  - Dashboard: dataset overview, feature histograms, model comparison chart

Run:
    streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_CANDIDATES = ["Housing (1).csv", "housing.csv"]
MODEL_PATH = "best_model.pkl"
METRICS_PATH = "model_metrics.csv"

NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


model = load_model()
df = load_data()
metrics_df = load_metrics()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.markdown(
    "Predict California median house values using a machine-learning pipeline "
    "trained on the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)."
)

if model is None:
    st.error(
        "**Model not found.** Please run `python train_model.py` first to train and save the model.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_dashboard = st.tabs(["🔮 Predict", "📊 Dashboard"])

# ===========================================================================
# TAB 1 — PREDICT
# ===========================================================================
with tab_predict:
    st.subheader("Enter House Features")
    st.markdown("Adjust the sliders and dropdowns below, then click **Predict Price**.")

    # Compute dataset stats for slider ranges (fall back to sensible defaults)
    if df is not None:
        stats = df[NUMERIC_FEATURES].describe()
        def col_min(c):  return float(stats.loc["min", c])
        def col_max(c):  return float(stats.loc["max", c])
        def col_med(c):  return float(stats.loc["50%", c])
    else:
        def col_min(c):  return 0.0
        def col_max(c):  return 1000.0
        def col_med(c):  return 500.0

    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.slider(
            "Longitude", min_value=col_min("longitude"), max_value=col_max("longitude"),
            value=col_med("longitude"), step=0.01,
            help="Negative values — further west."
        )
        latitude = st.slider(
            "Latitude", min_value=col_min("latitude"), max_value=col_max("latitude"),
            value=col_med("latitude"), step=0.01,
            help="Higher values — further north."
        )
        housing_median_age = st.slider(
            "Housing Median Age (years)",
            min_value=int(col_min("housing_median_age")),
            max_value=int(col_max("housing_median_age")),
            value=int(col_med("housing_median_age")),
        )

    with col2:
        total_rooms = st.number_input(
            "Total Rooms", min_value=1.0, max_value=col_max("total_rooms"),
            value=col_med("total_rooms"), step=10.0,
        )
        total_bedrooms = st.number_input(
            "Total Bedrooms", min_value=1.0, max_value=col_max("total_bedrooms"),
            value=col_med("total_bedrooms"), step=5.0,
        )
        population = st.number_input(
            "Population", min_value=1.0, max_value=col_max("population"),
            value=col_med("population"), step=10.0,
        )

    with col3:
        households = st.number_input(
            "Households", min_value=1.0, max_value=col_max("households"),
            value=col_med("households"), step=5.0,
        )
        median_income = st.slider(
            "Median Income (tens of thousands $)",
            min_value=col_min("median_income"),
            max_value=col_max("median_income"),
            value=col_med("median_income"),
            step=0.1,
        )
        ocean_proximity = st.selectbox("Ocean Proximity", options=OCEAN_OPTIONS)

    st.divider()

    if st.button("🔮 Predict Price", type="primary", disabled=model is None):
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": float(housing_median_age),
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }])

        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction complete!")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Predicted Median House Value", f"${prediction:,.0f}")
        col_res2.metric("Ocean Proximity", ocean_proximity)
        col_res3.metric("Median Income Band", f"${median_income * 10_000:,.0f} / yr")

        # Confidence context
        if df is not None:
            pct = (df["median_house_value"] < prediction).mean() * 100
            st.info(
                f"This predicted value is higher than **{pct:.1f}%** of all houses in the training dataset.",
                icon="ℹ️",
            )

# ===========================================================================
# TAB 2 — DASHBOARD
# ===========================================================================
with tab_dashboard:
    if df is None:
        st.warning("Dataset not found. Place `housing.csv` or `Housing (1).csv` in the project directory.")
    else:
        # ---- Dataset overview -----------------------------------------------
        st.subheader("📋 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", f"{df.shape[0]:,}")
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        # ---- Feature distributions ------------------------------------------
        st.subheader("📈 Feature Distributions")

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        for i, col in enumerate(NUMERIC_FEATURES):
            sns.histplot(df[col].dropna(), ax=axes[i], kde=True, color="#3b82d4", bins=40)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("")
        plt.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Target distribution
        st.markdown("**Target: Median House Value**")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        sns.histplot(df["median_house_value"].dropna(), kde=True, color="#7c5cd8", bins=50, ax=ax2)
        ax2.set_xlabel("Median House Value ($)")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()

        # Ocean proximity distribution
        st.subheader("🌊 Ocean Proximity Distribution")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ocean_counts = df["ocean_proximity"].value_counts()
        sns.barplot(x=ocean_counts.index, y=ocean_counts.values, ax=ax3, color="#3b82d4")
        ax3.set_xlabel("Ocean Proximity")
        ax3.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.divider()

        # ---- Model comparison -----------------------------------------------
        st.subheader("🤖 Model Comparison (5-Fold CV)")

        if metrics_df is None:
            st.info(
                "No model metrics found. Run `python train_model.py` to generate `model_metrics.csv`.",
                icon="ℹ️",
            )
        else:
            st.dataframe(
                metrics_df.style.format({"mean_r2": "{:.4f}", "mean_rmse": "{:,.0f}"}),
                use_container_width=True,
            )

            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))

            # R² bar chart
            colors = ["#3b82d4" if m != metrics_df.loc[metrics_df["mean_r2"].idxmax(), "model"]
                      else "#22c55e" for m in metrics_df["model"]]
            ax4a.barh(metrics_df["model"], metrics_df["mean_r2"], color=colors)
            ax4a.set_xlabel("Mean R² Score")
            ax4a.set_title("R² by Model (higher = better)")
            ax4a.set_xlim(0, 1)
            for i, v in enumerate(metrics_df["mean_r2"]):
                ax4a.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

            # RMSE bar chart
            ax4b.barh(metrics_df["model"], metrics_df["mean_rmse"], color=colors)
            ax4b.set_xlabel("Mean RMSE ($)")
            ax4b.set_title("RMSE by Model (lower = better)")
            for i, v in enumerate(metrics_df["mean_rmse"]):
                ax4b.text(v + 100, i, f"{v:,.0f}", va="center", fontsize=9)

            plt.suptitle("Cross-Validation Model Comparison", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            best_model_name = metrics_df.loc[metrics_df["mean_r2"].idxmax(), "model"]
            best_r2 = metrics_df["mean_r2"].max()
            st.success(f"✅ **Best model selected:** {best_model_name} (CV R² = {best_r2:.4f})", icon="🏆")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#57606a; font-size:12px;'>"
    "House Price Predictor · Built with Streamlit &amp; scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
