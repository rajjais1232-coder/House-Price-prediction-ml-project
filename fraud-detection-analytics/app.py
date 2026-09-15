"""
app.py — Fraud Detection & Transaction Analytics Dashboard
===========================================================
Streamlit application — run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import warnings
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup so src/ imports work regardless of cwd ───────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_processing import (
    detect_column_mapping,
    apply_column_mapping,
    dataset_quality_report,
    clean_dataset,
    feature_summary,
    TARGET_COL,
)
from src.analytics import (
    calculate_kpis,
    transactions_by_day,
    transactions_by_month,
    fraud_by_hour,
    customer_summary,
    high_risk_customers,
    merchant_summary,
    fraud_by_merchant_category,
    fraud_by_location,
    fraud_by_column,
    international_fraud_comparison,
)
from src.feature_engineering import (
    load_artifacts,
    prepare_single_transaction,
    FEATURE_CANDIDATES,
    ENCODE_COLS,
)
from src.model_evaluation import (
    evaluate_model,
    get_confusion_matrix_fig,
    get_feature_importance_fig,
    get_roc_pr_curves,
    categorise_risk,
    risk_color,
)

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Styling
# ============================================================================
st.markdown("""
<style>
    .kpi-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-label { font-size: 13px; color: #57606a; font-weight: 500; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #1f2328; margin-top: 4px; }
    .kpi-value.fraud { color: #d73a49; }
    .kpi-value.ok    { color: #2da44e; }
    section[data-testid="stSidebar"] { background: #1f2328; }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .st-emotion-cache-1dp5vir { background: #3b82d4; }
    h1, h2, h3 { font-family: "Segoe UI", system-ui, sans-serif; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helpers
# ============================================================================

def fmt_currency(v) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.2f}"


def fmt_pct(v) -> str:
    return f"{v:.2f}%" if v is not None else "N/A"


def fmt_num(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:,}"


def kpi_card(label: str, value: str, style: str = "") -> str:
    cls = f"kpi-value {style}"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>"""


def csv_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=f"⬇ {label}", data=csv, file_name=filename, mime="text/csv")


# ============================================================================
# Session state keys
# ============================================================================
SS_DF_CLEAN    = "df_clean"
SS_DF_RAW      = "df_raw"
SS_MAPPING     = "col_mapping"
SS_ARTIFACTS   = "ml_artifacts"
SS_TRAINED     = "trained_models"
SS_COMPARISON  = "model_comparison"
SS_BEST        = "best_model_name"
SS_ANOMALY_DF  = "anomaly_df"


# ============================================================================
# Data loading — cached
# ============================================================================

@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    sample_path = os.path.join(ROOT, "data", "sample_transactions.csv")
    return pd.read_csv(sample_path)


@st.cache_resource(show_spinner=False)
def load_saved_model(path: str):
    try:
        return load_artifacts(path)
    except Exception:
        return None


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🔍 Fraud Analytics")
        st.markdown("---")

        # Upload
        uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
        if st.button("📊 Use Sample Dataset", use_container_width=True):
            st.session_state["use_sample"] = True
            st.session_state[SS_DF_RAW] = None
            st.session_state[SS_DF_CLEAN] = None

        st.markdown("---")
        st.markdown("### 🎛 Filters")

        filters = {}

        # Date range
        if "Transaction_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Transaction_Date"]):
            min_d = df["Transaction_Date"].min().date()
            max_d = df["Transaction_Date"].max().date()
            date_range = st.date_input("📅 Date Range", value=(min_d, max_d))
            filters["date_range"] = date_range

        # Categorical filters
        for col, label in [
            ("Merchant_Category", "🏪 Merchant Category"),
            ("Location",          "📍 Location"),
            ("Payment_Method",    "💳 Payment Method"),
            ("Transaction_Type",  "🔄 Transaction Type"),
        ]:
            if col in df.columns:
                opts = sorted(df[col].dropna().unique().tolist())
                sel = st.multiselect(label, opts, default=[], key=f"filter_{col}")
                if sel:
                    filters[col] = sel

        # Fraud filter
        if "Fraud" in df.columns:
            fraud_opts = {"All": None, "Fraud Only": 1, "Legitimate Only": 0}
            fraud_sel = st.selectbox("⚠️ Fraud Status", list(fraud_opts.keys()))
            filters["Fraud"] = fraud_opts[fraud_sel]

        st.markdown("---")
        st.markdown("### 📑 Navigation")
        pages = [
            "📊 Executive Overview",
            "🚨 Fraud Analysis",
            "👤 Customer Risk Analytics",
            "🏪 Merchant Analytics",
            "🤖 ML Fraud Prediction",
        ]
        page = st.radio("Go to", pages, label_visibility="collapsed")

    return uploaded, page, filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to the DataFrame."""
    df = df.copy()

    if "date_range" in filters:
        dr = filters["date_range"]
        if len(dr) == 2 and "Transaction_Date" in df.columns:
            df = df[
                (df["Transaction_Date"].dt.date >= dr[0]) &
                (df["Transaction_Date"].dt.date <= dr[1])
            ]

    for col in ["Merchant_Category", "Location", "Payment_Method", "Transaction_Type"]:
        if col in filters and filters[col]:
            df = df[df[col].isin(filters[col])]

    if "Fraud" in filters and filters["Fraud"] is not None:
        df = df[df["Fraud"] == filters["Fraud"]]

    return df


# ============================================================================
# PAGE 1 — Executive Overview
# ============================================================================

def page_executive_overview(df: pd.DataFrame):
    st.title("📊 Executive Overview")
    st.markdown("High-level KPIs and transaction trends across the filtered dataset.")
    st.markdown("---")

    kpis = calculate_kpis(df)
    has_fraud = "Fraud" in df.columns

    # KPI row 1
    cols = st.columns(3)
    cols[0].markdown(kpi_card("Total Transactions",    fmt_num(kpis["Total Transactions"]), "ok"), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Total Amount",          fmt_currency(kpis["Total Amount"]), "ok"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Avg Transaction Amount",fmt_currency(kpis["Avg Transaction Amount"])), unsafe_allow_html=True)

    # KPI row 2
    cols2 = st.columns(3)
    cols2[0].markdown(kpi_card("Fraud Transactions",   fmt_num(kpis["Fraud Transactions"]),  "fraud"), unsafe_allow_html=True)
    cols2[1].markdown(kpi_card("Fraud Rate",           fmt_pct(kpis["Fraud Rate (%)"]),       "fraud"), unsafe_allow_html=True)
    cols2[2].markdown(kpi_card("Fraud Amount",         fmt_currency(kpis["Fraud Amount"]),    "fraud"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Transaction volume over time ─────────────────────────────────────────
    daily = transactions_by_day(df)
    if not daily.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.line(
                daily, x="Transaction_Day", y="Transactions",
                title="📈 Transaction Volume Over Time",
                color_discrete_sequence=["#3b82d4"],
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Transactions", height=320)
            st.plotly_chart(fig, use_container_width=True)

        if has_fraud:
            with col_b:
                fig2 = px.line(
                    daily, x="Transaction_Day", y="Fraud_Count",
                    title="🚨 Fraud Trend Over Time",
                    color_discrete_sequence=["#d73a49"],
                )
                fig2.update_layout(xaxis_title="Date", yaxis_title="Fraud Count", height=320)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Fraud vs Legitimate pie ───────────────────────────────────────────────
    if has_fraud:
        col_c, col_d, col_e = st.columns(3)
        fraud_counts = df["Fraud"].value_counts().rename({0: "Legitimate", 1: "Fraudulent"})

        with col_c:
            fig = px.pie(
                names=fraud_counts.index,
                values=fraud_counts.values,
                title="Fraud vs Legitimate",
                color=fraud_counts.index,
                color_discrete_map={"Fraudulent": "#d73a49", "Legitimate": "#2da44e"},
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        # By payment method
        if "Payment_Method" in df.columns:
            pm = fraud_by_column(df, "Payment_Method").head(6)
            with col_d:
                fig = px.bar(
                    pm, x="Payment_Method", y="Fraud_Rate",
                    title="💳 Fraud Rate by Payment Method",
                    color="Fraud_Rate",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(xaxis_title="", height=320, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # By merchant category
        if "Merchant_Category" in df.columns:
            mc = fraud_by_merchant_category(df).head(8)
            with col_e:
                fig = px.bar(
                    mc, x="Merchant_Category", y="Fraud_Rate",
                    title="🏪 Fraud Rate by Merchant Category",
                    color="Fraud_Rate",
                    color_continuous_scale="Oranges",
                )
                fig.update_layout(xaxis_title="", height=320, showlegend=False,
                                  xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

    # ── Fraud by Location ────────────────────────────────────────────────────
    if has_fraud and "Location" in df.columns:
        loc = fraud_by_location(df).head(15)
        fig = px.bar(
            loc, x="Fraud_Rate", y="Location",
            orientation="h",
            title="📍 Fraud Rate by Location (Top 15)",
            color="Fraud_Rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=460, yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_download(df, "Download Filtered Transactions", "filtered_transactions.csv")
    with col_dl2:
        if has_fraud:
            fraud_df = df[df["Fraud"] == 1]
            csv_download(fraud_df, "Download Fraudulent Transactions", "fraudulent_transactions.csv")


# ============================================================================
# PAGE 2 — Fraud Analysis
# ============================================================================

def page_fraud_analysis(df: pd.DataFrame):
    st.title("🚨 Fraud Analysis")
    st.markdown("Deep-dive into fraud patterns across time, device, and transaction dimensions.")
    st.markdown("---")

    has_fraud = "Fraud" in df.columns
    if not has_fraud:
        st.warning("⚠️ No 'Fraud' column detected. Upload a labelled dataset to use this page.")
        return

    # ── Row 1: Fraud by Hour | Fraud by Transaction Type ────────────────────
    col_a, col_b = st.columns(2)
    hr = fraud_by_hour(df)
    if not hr.empty:
        with col_a:
            fig = px.bar(
                hr, x="Transaction_Hour", y="Fraud_Rate",
                title="🕐 Fraud Rate by Hour of Day",
                color="Fraud_Rate",
                color_continuous_scale="Reds",
            )
            fig.update_layout(xaxis_title="Hour", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if "Transaction_Type" in df.columns:
        tt = fraud_by_column(df, "Transaction_Type")
        with col_b:
            fig = px.bar(
                tt, x="Transaction_Type", y="Fraud_Rate",
                title="🔄 Fraud Rate by Transaction Type",
                color="Fraud_Rate",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(xaxis_title="", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Device | Amount Distribution ─────────────────────────────────
    col_c, col_d = st.columns(2)
    if "Device_Type" in df.columns:
        dev = fraud_by_column(df, "Device_Type")
        with col_c:
            fig = px.pie(
                dev, names="Device_Type", values="Fraud_Count",
                title="📱 Fraud Count by Device",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        fraud_amounts = df[df["Fraud"] == 1]["Amount"]
        legit_amounts = df[df["Fraud"] == 0]["Amount"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_amounts, name="Legitimate", opacity=0.6,
                                   marker_color="#2da44e", nbinsx=50))
        fig.add_trace(go.Histogram(x=fraud_amounts, name="Fraudulent", opacity=0.7,
                                   marker_color="#d73a49", nbinsx=50))
        fig.update_layout(
            barmode="overlay",
            title="💰 Transaction Amount Distribution: Fraud vs Legitimate",
            xaxis_title="Amount ($)",
            yaxis_title="Count",
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: International | Fraud by Merchant ─────────────────────────────
    col_e, col_f = st.columns(2)
    intl = international_fraud_comparison(df)
    if not intl.empty:
        with col_e:
            fig = px.bar(
                intl, x="International_Transaction", y="Fraud_Rate",
                title="🌍 International vs Domestic Fraud Rate",
                color="International_Transaction",
                color_discrete_map={"Yes": "#d73a49", "No": "#3b82d4"},
            )
            fig.update_layout(xaxis_title="", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if "Merchant" in df.columns:
        merch = merchant_summary(df)
        if not merch.empty and "Fraud_Count" in merch.columns:
            top_merch = merch.nlargest(10, "Fraud_Count")
            with col_f:
                fig = px.bar(
                    top_merch, x="Fraud_Count", y="Merchant",
                    orientation="h",
                    title="🏪 Top 10 Merchants by Fraud Count",
                    color="Fraud_Count",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(yaxis_title="", height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── Monthly fraud amount ─────────────────────────────────────────────────
    monthly = transactions_by_month(df)
    if not monthly.empty and "Fraud_Count" in monthly.columns:
        fig = px.area(
            monthly, x="Transaction_Month", y="Fraud_Count",
            title="📅 Monthly Fraud Count",
            color_discrete_sequence=["#d73a49"],
        )
        fig.update_layout(height=300, xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)

    # ── Top suspicious transactions table ────────────────────────────────────
    st.markdown("### 🔎 Top High-Value Fraudulent Transactions")
    show_cols = [c for c in ["Transaction_ID", "Customer_ID", "Transaction_Date", "Amount",
                              "Merchant", "Location", "Payment_Method", "Device_Type", "Fraud"]
                 if c in df.columns]
    top_fraud = df[df["Fraud"] == 1].nlargest(20, "Amount")[show_cols]
    st.dataframe(top_fraud, use_container_width=True)


# ============================================================================
# PAGE 3 — Customer Risk Analytics
# ============================================================================

def page_customer_risk(df: pd.DataFrame):
    st.title("👤 Customer Risk Analytics")
    st.markdown("Customer-level fraud exposure, high-risk profiling, and segment analysis.")
    st.markdown("---")

    has_fraud = "Fraud" in df.columns
    if "Customer_ID" not in df.columns:
        st.warning("⚠️ No 'Customer_ID' column detected.")
        return

    cust = customer_summary(df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Unique Customers",   fmt_num(len(cust))), unsafe_allow_html=True)
    if has_fraud and "Fraud_Count" in cust.columns:
        c2.markdown(kpi_card("Customers with Fraud",
                             fmt_num((cust["Fraud_Count"] > 0).sum()), "fraud"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Avg Fraud Rate",
                             fmt_pct(cust["Fraud_Rate"].mean()), "fraud"), unsafe_allow_html=True)
        c4.markdown(kpi_card("Max Fraud Rate (1 Customer)",
                             fmt_pct(cust["Fraud_Rate"].max()), "fraud"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Fraud by Customer Age ─────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    if "Customer_Age" in df.columns and has_fraud:
        bins = [18, 25, 35, 45, 55, 65, 100]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        df_tmp = df.copy()
        df_tmp["Age_Group"] = pd.cut(df_tmp["Customer_Age"], bins=bins, labels=labels, right=False)
        age_grp = fraud_by_column(df_tmp, "Age_Group")
        with col_a:
            fig = px.bar(
                age_grp, x="Age_Group", y="Fraud_Rate",
                title="👥 Fraud Rate by Customer Age Group",
                color="Fraud_Rate",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(xaxis_title="Age Group", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Fraud rate by account age ─────────────────────────────────────────────
    if "Account_Age" in df.columns and has_fraud:
        bins2 = [0, 6, 12, 24, 60, 999]
        labels2 = ["<6m", "6-12m", "1-2yr", "2-5yr", "5+yr"]
        df_tmp2 = df.copy()
        df_tmp2["Account_Tenure"] = pd.cut(df_tmp2["Account_Age"], bins=bins2, labels=labels2, right=False)
        acc_grp = fraud_by_column(df_tmp2, "Account_Tenure")
        with col_b:
            fig = px.bar(
                acc_grp, x="Account_Tenure", y="Fraud_Rate",
                title="📅 Fraud Rate by Account Tenure",
                color="Fraud_Rate",
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_title="Account Age", height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Top customers scatter (amount vs fraud rate) ─────────────────────────
    if has_fraud and "Fraud_Rate" in cust.columns:
        fig = px.scatter(
            cust,
            x="Total_Amount",
            y="Fraud_Rate",
            size="Total_Transactions",
            color="Fraud_Rate",
            hover_data=["Customer_ID", "Fraud_Count"],
            title="💰 Customer Total Amount vs Fraud Rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── High-Risk Customer Table ─────────────────────────────────────────────
    st.markdown("### 🚨 High-Risk Customers (Fraud Rate ≥ 30%, Min 3 Transactions)")
    hr_cust = high_risk_customers(df)
    if hr_cust.empty:
        st.info("No customers meet the high-risk threshold in the current filter.")
    else:
        st.dataframe(hr_cust.reset_index(drop=True), use_container_width=True)
        csv_download(hr_cust, "Download High-Risk Customers", "high_risk_customers.csv")

    # ── Failed transactions ─────────────────────────────────────────────────
    if "Failed_Transactions" in df.columns and has_fraud:
        st.markdown("### ❌ Failed Transaction Patterns")
        fail_bins = [0, 1, 3, 5, 10, 100]
        fail_labels = ["0", "1-2", "3-4", "5-9", "10+"]
        df_fail = df.copy()
        df_fail["Fail_Group"] = pd.cut(df_fail["Failed_Transactions"], bins=fail_bins, labels=fail_labels, right=False)
        fail_grp = fraud_by_column(df_fail, "Fail_Group")
        fig = px.bar(
            fail_grp, x="Fail_Group", y="Fraud_Rate",
            title="Failed Transactions vs Fraud Rate",
            color="Fraud_Rate",
            color_continuous_scale="Reds",
        )
        fig.update_layout(xaxis_title="Failed Transactions", height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Full customer table ──────────────────────────────────────────────────
    with st.expander("📋 Full Customer Summary Table"):
        st.dataframe(cust.reset_index(drop=True), use_container_width=True)
        csv_download(cust, "Download Customer Summary", "customer_summary.csv")


# ============================================================================
# PAGE 4 — Merchant Analytics
# ============================================================================

def page_merchant_analytics(df: pd.DataFrame):
    st.title("🏪 Merchant Analytics")
    st.markdown("Merchant-level fraud exposure, top fraud merchants, and category analysis.")
    st.markdown("---")

    if "Merchant" not in df.columns:
        st.warning("⚠️ No 'Merchant' column detected.")
        return

    has_fraud = "Fraud" in df.columns
    merch = merchant_summary(df)
    cat   = fraud_by_merchant_category(df)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Unique Merchants", fmt_num(merch["Merchant"].nunique() if not merch.empty else 0)), unsafe_allow_html=True)
    if has_fraud and "Fraud_Count" in merch.columns:
        c2.markdown(kpi_card("Merchants with Fraud",
                             fmt_num((merch["Fraud_Count"] > 0).sum()), "fraud"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Max Merchant Fraud Rate",
                             fmt_pct(merch["Fraud_Rate"].max() if "Fraud_Rate" in merch.columns else None), "fraud"),
                    unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Top by transaction volume
    top_vol = merch.nlargest(10, "Total_Transactions")
    with col_a:
        fig = px.bar(
            top_vol, x="Total_Transactions", y="Merchant",
            orientation="h",
            title="📊 Top 10 Merchants by Transaction Volume",
            color="Total_Transactions",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis_title="", height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Top by fraud count
    if has_fraud and "Fraud_Count" in merch.columns:
        top_fraud = merch.nlargest(10, "Fraud_Count")
        with col_b:
            fig = px.bar(
                top_fraud, x="Fraud_Count", y="Merchant",
                orientation="h",
                title="🚨 Top 10 Merchants by Fraud Count",
                color="Fraud_Count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(yaxis_title="", height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Fraud by amount
    if has_fraud and "Fraud_Amount" in merch.columns:
        top_amt = merch.dropna(subset=["Fraud_Amount"]).nlargest(10, "Fraud_Amount")
        with col_c:
            fig = px.bar(
                top_amt, x="Fraud_Amount", y="Merchant",
                orientation="h",
                title="💸 Top 10 Merchants by Fraud Amount",
                color="Fraud_Amount",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(yaxis_title="", height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Fraud rate by category
    if not cat.empty:
        with col_d:
            fig = px.bar(
                cat, x="Merchant_Category", y="Fraud_Rate",
                title="🏷️ Fraud Rate by Merchant Category",
                color="Fraud_Rate",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(xaxis_title="", height=370, showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    # Table
    with st.expander("📋 Full Merchant Summary Table"):
        display_cols = [c for c in ["Merchant", "Merchant_Category", "Total_Transactions",
                                     "Total_Amount", "Avg_Amount", "Fraud_Count",
                                     "Fraud_Rate", "Fraud_Amount"] if c in merch.columns]
        st.dataframe(merch[display_cols].reset_index(drop=True), use_container_width=True)
        csv_download(merch[display_cols], "Download Merchant Analytics", "merchant_analytics.csv")


# ============================================================================
# PAGE 5 — ML Fraud Prediction
# ============================================================================

def page_ml_prediction(df: pd.DataFrame):
    st.title("🤖 ML Fraud Prediction")
    st.markdown("Train ML models, compare performance, and predict fraud probability.")
    st.markdown("---")

    has_fraud = "Fraud" in df.columns

    if not has_fraud:
        st.error("❌ No 'Fraud' column found. Supervised ML requires a labelled target column.")
        st.info("💡 You can still use the analytics pages to explore your data.")
        return

    class_counts = df["Fraud"].value_counts()
    fraud_count  = int(class_counts.get(1, 0))
    legit_count  = int(class_counts.get(0, 0))
    fraud_rate   = fraud_count / len(df) * 100

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Training Samples", fmt_num(len(df))), unsafe_allow_html=True)
    c2.markdown(kpi_card("Fraud Samples", fmt_num(fraud_count), "fraud"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Fraud Rate", fmt_pct(fraud_rate), "fraud"), unsafe_allow_html=True)

    if fraud_count < 10:
        st.warning("⚠️ Very few fraud samples. Model performance may be unreliable.")

    st.markdown("---")

    # ── Model Training ───────────────────────────────────────────────────────
    st.markdown("### 🏋️ Model Training")
    st.markdown("""
    Train and compare **Logistic Regression**, **Decision Tree**, and **Random Forest**.
    Models are evaluated using **Precision, Recall, F1, and ROC-AUC**
    (NOT accuracy — fraud datasets are imbalanced).
    """)

    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        with st.spinner("Training models... this may take a moment."):
            try:
                from src.model_training import train_all_models
                trained, comparison, artifacts, best_name, feat_names = train_all_models(df)
                st.session_state[SS_TRAINED]    = trained
                st.session_state[SS_COMPARISON] = comparison
                st.session_state[SS_ARTIFACTS]  = artifacts
                st.session_state[SS_BEST]       = best_name
                st.success(f"✅ Training complete! Best model: **{best_name}** (by F1 Score)")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                return

    # Show results if available
    if SS_COMPARISON in st.session_state and st.session_state[SS_COMPARISON] is not None:
        comp_df = st.session_state[SS_COMPARISON]
        trained = st.session_state.get(SS_TRAINED, {})
        artifacts = st.session_state[SS_ARTIFACTS]
        best_name = st.session_state[SS_BEST]

        st.markdown("### 📊 Model Comparison")
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["Precision", "Recall", "F1 Score", "ROC-AUC"],
                color="#c8e6c9",
                axis=0,
            ),
            use_container_width=True,
        )
        csv_download(comp_df, "Download Model Comparison", "model_comparison.csv")

        st.markdown(f"**🏆 Best Model: {best_name}** — selected by highest F1 Score")

        # ── Confusion Matrix ────────────────────────────────────────────────
        st.markdown("### 🔢 Confusion Matrix")
        X_test = artifacts.get("X_test")
        y_test = artifacts.get("y_test")
        best_model = artifacts.get("model")

        if X_test is not None and y_test is not None and best_model is not None:
            cm_fig = get_confusion_matrix_fig(best_model, X_test, y_test, best_name)
            st.plotly_chart(cm_fig, use_container_width=True)

        # ── ROC + PR curves ─────────────────────────────────────────────────
        if trained and X_test is not None:
            st.markdown("### 📈 ROC & Precision-Recall Curves")
            roc_fig, pr_fig = get_roc_pr_curves(trained, X_test, y_test)
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(roc_fig, use_container_width=True)
            with cb:
                st.plotly_chart(pr_fig, use_container_width=True)

        # ── Feature Importance ──────────────────────────────────────────────
        feat_names = artifacts.get("feature_names", [])
        fi_fig = get_feature_importance_fig(best_model, feat_names, best_name)
        if fi_fig:
            st.markdown("### 🔍 Feature Importance")
            st.plotly_chart(fi_fig, use_container_width=True)

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕵️ Anomaly Detection (Isolation Forest)")
    st.markdown("Identify unusual transactions based on numerical feature patterns.")

    contamination = st.slider("Anomaly contamination rate", 0.01, 0.20, 0.05, 0.01)
    if st.button("🔍 Run Anomaly Detection", use_container_width=True):
        with st.spinner("Running Isolation Forest..."):
            try:
                from src.model_training import detect_anomalies
                anomaly_df = detect_anomalies(df, contamination=contamination)
                st.session_state[SS_ANOMALY_DF] = anomaly_df
                n_anomalies = (anomaly_df["Anomaly"] == -1).sum()
                st.success(f"✅ Found {n_anomalies} anomalies ({n_anomalies/len(anomaly_df)*100:.1f}%)")
            except Exception as e:
                st.error(f"❌ Anomaly detection failed: {e}")

    if SS_ANOMALY_DF in st.session_state and st.session_state[SS_ANOMALY_DF] is not None:
        andf = st.session_state[SS_ANOMALY_DF]
        suspicious = andf[andf["Anomaly"] == -1].sort_values("Anomaly_Score").head(20)
        show_cols = [c for c in ["Transaction_ID", "Customer_ID", "Amount", "Failed_Transactions",
                                  "Account_Age", "Anomaly_Score", "Fraud"] if c in suspicious.columns]
        st.markdown("#### 🚨 Top Suspicious Transactions")
        st.dataframe(suspicious[show_cols].reset_index(drop=True), use_container_width=True)
        csv_download(suspicious[show_cols], "Download Anomalies", "anomalies.csv")

    # ── Interactive Prediction Form ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Fraud Probability Prediction")
    st.markdown("Enter transaction details to predict the probability of fraud.")

    artifacts = st.session_state.get(SS_ARTIFACTS)
    model_path = os.path.join(ROOT, "models", "fraud_model.pkl")

    if artifacts is None:
        saved = load_saved_model(model_path)
        if saved:
            artifacts = saved
            st.info("📂 Using previously saved model.")
        else:
            st.info("⚠️ Train the models above first to enable predictions.")
            return

    with st.form("prediction_form"):
        st.markdown("**Transaction Details**")
        col1, col2, col3 = st.columns(3)

        with col1:
            amount      = st.number_input("Amount ($)",             min_value=0.0, value=150.0, step=1.0)
            cust_age    = st.number_input("Customer Age",           min_value=18, max_value=100, value=35)
            acct_age    = st.number_input("Account Age (months)",   min_value=0, max_value=360, value=24)
            failed_txns = st.number_input("Failed Transactions",    min_value=0, max_value=100, value=0)

        with col2:
            prev_txns   = st.number_input("Previous Transactions",  min_value=0, value=50)
            txn_freq    = st.number_input("Transaction Frequency",  min_value=1, value=10)
            international = st.selectbox("International?",          ["No", "Yes"])
            device      = st.selectbox("Device Type",
                                       ["Mobile", "Desktop", "Tablet", "POS Terminal", "ATM"])

        with col3:
            payment     = st.selectbox("Payment Method",
                                       ["Credit Card", "Debit Card", "PayPal",
                                        "Bank Transfer", "Crypto", "Gift Card"])
            txn_type    = st.selectbox("Transaction Type",
                                       ["Purchase", "Transfer", "Withdrawal", "Refund", "Subscription"])
            merch_cat   = st.selectbox("Merchant Category",
                                       ["Retail", "E-commerce", "Food & Beverage", "Electronics",
                                        "Gas & Fuel", "Healthcare", "Travel", "Money Transfer",
                                        "Subscription", "ATM", "Unknown"])
            location    = st.selectbox("Location",
                                       ["New York, NY", "Los Angeles, CA", "Chicago, IL",
                                        "Houston, TX", "Miami, FL", "International - UK",
                                        "International - Unknown"])

        submitted = st.form_submit_button("🔮 Predict Fraud Probability", type="primary", use_container_width=True)

    if submitted:
        txn_dict = {
            "Amount":                   amount,
            "Customer_Age":             cust_age,
            "Account_Age":              acct_age,
            "Failed_Transactions":      failed_txns,
            "Previous_Transactions":    prev_txns,
            "Transaction_Frequency":    txn_freq,
            "International_Transaction": international,
            "Device_Type":              device,
            "Payment_Method":           payment,
            "Transaction_Type":         txn_type,
            "Merchant_Category":        merch_cat,
            "Location":                 location,
        }
        try:
            model        = artifacts["model"]
            encoders     = artifacts["encoders"]
            scaler       = artifacts["scaler"]
            feature_names = artifacts["feature_names"]

            X_pred = prepare_single_transaction(txn_dict, encoders, scaler, feature_names)
            prob   = model.predict_proba(X_pred)[0][1]
            risk   = categorise_risk(prob)
            pred   = "🚨 FRAUD" if prob >= 0.5 else "✅ LEGITIMATE"
            color  = risk_color(risk)

            st.markdown("---")
            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.markdown(kpi_card("Prediction", pred, "fraud" if prob >= 0.5 else "ok"),
                            unsafe_allow_html=True)
            res_c2.markdown(kpi_card("Fraud Probability", f"{prob*100:.1f}%",
                                     "fraud" if prob >= 0.5 else "ok"), unsafe_allow_html=True)
            res_c3.markdown(kpi_card("Risk Level", risk), unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  30], "color": "#d4edda"},
                        {"range": [30, 70], "color": "#fff3cd"},
                        {"range": [70, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": prob * 100},
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk factors explanation
            st.markdown("#### ⚠️ Key Risk Factors")
            risk_factors = []
            if international == "Yes":
                risk_factors.append("🌍 International transaction")
            if amount > 2000:
                risk_factors.append(f"💰 High transaction amount (${amount:,.2f})")
            if failed_txns > 5:
                risk_factors.append(f"❌ High failed transactions ({failed_txns})")
            if payment in ("Crypto", "Gift Card"):
                risk_factors.append(f"💳 High-risk payment method ({payment})")
            if merch_cat in ("Money Transfer", "ATM", "Unknown"):
                risk_factors.append(f"🏪 High-risk merchant category ({merch_cat})")
            if acct_age < 6:
                risk_factors.append(f"📅 New account (account age: {acct_age} months)")
            if txn_type in ("Transfer", "Withdrawal"):
                risk_factors.append(f"🔄 High-risk transaction type ({txn_type})")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.markdown("- No major risk flags detected.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ============================================================================
# Data Processing Page (shown as sidebar modal after upload)
# ============================================================================

def page_data_processing(raw_df: pd.DataFrame):
    """Show data quality report and allow column mapping confirmation."""
    st.title("📥 Data Processing & Quality Report")
    st.markdown("---")

    mapping = detect_column_mapping(raw_df)

    with st.expander("🗂 Column Mapping (auto-detected)", expanded=True):
        mapping_display = pd.DataFrame(
            [(k, v if v else "❌ Not Found") for k, v in mapping.items()],
            columns=["Expected Column", "Detected As"],
        )
        st.dataframe(mapping_display, use_container_width=True)

    df_mapped = apply_column_mapping(raw_df, mapping)
    df_clean, actions = clean_dataset(df_mapped)
    report = dataset_quality_report(df_mapped)

    # Quality KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Rows",     fmt_num(report["total_rows"])),     unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Columns",  fmt_num(report["total_columns"])),  unsafe_allow_html=True)
    c3.markdown(kpi_card("Duplicate Rows", fmt_num(report["duplicate_rows"])), unsafe_allow_html=True)
    c4.markdown(kpi_card("Missing Cells",  fmt_num(report["total_missing_cells"])), unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Missing values
    with col_a:
        st.markdown("#### Missing Values by Column")
        missing = pd.DataFrame({
            "Column":  list(report["missing_by_column"].keys()),
            "Missing": list(report["missing_by_column"].values()),
            "Missing %": list(report["missing_pct_by_column"].values()),
        }).query("Missing > 0")
        if missing.empty:
            st.success("✅ No missing values detected!")
        else:
            fig = px.bar(missing, x="Column", y="Missing %",
                         color="Missing %", color_continuous_scale="Reds")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Outliers
    with col_b:
        st.markdown("#### Outliers by Column (IQR Method)")
        outliers = pd.DataFrame({
            "Column": list(report["outlier_counts"].keys()),
            "Outliers": list(report["outlier_counts"].values()),
        }).query("Outliers > 0")
        if outliers.empty:
            st.success("✅ No outliers detected!")
        else:
            fig = px.bar(outliers, x="Column", y="Outliers",
                         color="Outliers", color_continuous_scale="Oranges")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Class balance
    if report["class_balance"]:
        st.markdown("#### Class Balance (Fraud / Legitimate)")
        cb = pd.DataFrame(
            [(k, v) for k, v in report["class_balance"].items()],
            columns=["Class", "Count"],
        )
        cb["Label"] = cb["Class"].map({0: "Legitimate", 1: "Fraudulent"})
        fig = px.pie(cb, names="Label", values="Count",
                     color="Label",
                     color_discrete_map={"Fraudulent": "#d73a49", "Legitimate": "#2da44e"})
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Fraud Rate:** {report['fraud_rate']}%")

    # Cleaning actions
    st.markdown("#### ✅ Cleaning Actions Applied")
    for action in actions:
        st.markdown(f"- {action}")

    # Feature summary
    with st.expander("📋 Feature Summary"):
        st.dataframe(feature_summary(df_clean), use_container_width=True)

    # Preview
    with st.expander("👀 Cleaned Dataset Preview (first 100 rows)"):
        st.dataframe(df_clean.head(100), use_container_width=True)

    # Save to session state
    st.session_state[SS_DF_CLEAN] = df_clean
    csv_download(df_clean, "Download Cleaned Dataset", "cleaned_transactions.csv")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    # ── Initial state ────────────────────────────────────────────────────────
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    raw_df = None

    # ── Sidebar (provides uploader + filters + nav) ──────────────────────────
    # We need a preliminary df for filter options; use empty df if none loaded
    preview_df = st.session_state.get(SS_DF_CLEAN, pd.DataFrame())
    uploaded, page, filters = render_sidebar(preview_df)

    # ── Load data ─────────────────────────────────────────────────────────────
    if uploaded is not None:
        try:
            raw_df = load_csv(uploaded.read())
            if raw_df.empty:
                st.error("❌ Uploaded CSV is empty.")
                return
            # If new upload, trigger processing
            if st.session_state.get("last_upload") != uploaded.name:
                st.session_state["last_upload"]  = uploaded.name
                st.session_state[SS_DF_CLEAN]    = None
                st.session_state[SS_DF_RAW]      = raw_df
                st.session_state[SS_ARTIFACTS]   = None
                st.session_state[SS_TRAINED]     = None
                st.session_state[SS_COMPARISON]  = None
                st.session_state[SS_ANOMALY_DF]  = None
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            return

    elif st.session_state.get("use_sample"):
        try:
            raw_df = load_sample()
            if st.session_state.get(SS_DF_RAW) is None:
                st.session_state[SS_DF_RAW]   = raw_df
                st.session_state[SS_DF_CLEAN] = None
        except Exception as e:
            st.error(f"❌ Failed to load sample dataset: {e}")
            return

    # ── Auto-process if raw data available but clean not yet computed ─────────
    if raw_df is not None and st.session_state.get(SS_DF_CLEAN) is None:
        mapping  = detect_column_mapping(raw_df)
        df_m     = apply_column_mapping(raw_df, mapping)
        df_clean, _ = clean_dataset(df_m)
        st.session_state[SS_DF_CLEAN] = df_clean

    # ── Landing page if no data loaded ──────────────────────────────────────
    if st.session_state.get(SS_DF_CLEAN) is None:
        _render_landing()
        return

    clean_df = st.session_state[SS_DF_CLEAN]

    # ── Apply sidebar filters ─────────────────────────────────────────────────
    df_filtered = apply_filters(clean_df, filters)
    if df_filtered.empty:
        st.warning("⚠️ No data matches the current filters. Try adjusting the sidebar filters.")
        return

    # ── Route to page ─────────────────────────────────────────────────────────
    if "Data Processing" in page or uploaded is not None and st.session_state.get("show_processing"):
        page_data_processing(st.session_state.get(SS_DF_RAW, clean_df))
    elif "Executive Overview" in page:
        page_executive_overview(df_filtered)
    elif "Fraud Analysis" in page:
        page_fraud_analysis(df_filtered)
    elif "Customer Risk" in page:
        page_customer_risk(df_filtered)
    elif "Merchant" in page:
        page_merchant_analytics(df_filtered)
    elif "ML Fraud" in page:
        page_ml_prediction(clean_df)  # ML uses full unfiltered clean data for training


def _render_landing():
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <h1>🔍 Fraud Detection & Transaction Analytics</h1>
        <p style="font-size:18px; color:#57606a;">
            A production-style analytics dashboard for financial fraud investigation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.info("📂 **Upload a CSV** transaction dataset using the sidebar to get started.")
    col2.info("📊 **Use the Sample Dataset** button to explore the full dashboard immediately.")
    col3.info("🤖 **Train ML models** to predict fraud probability for any transaction.")

    st.markdown("---")
    st.markdown("### 📋 Expected Dataset Columns")
    st.markdown("""
    | Column | Description |
    |---|---|
    | Transaction_ID | Unique transaction identifier |
    | Customer_ID | Customer identifier |
    | Transaction_Date | Date of transaction |
    | Amount | Transaction amount |
    | Merchant | Merchant name |
    | Merchant_Category | Merchant business category |
    | Location | Transaction location |
    | Transaction_Type | Purchase / Transfer / Withdrawal |
    | Payment_Method | Credit Card / Debit Card / etc |
    | Device_Type | Mobile / Desktop / ATM / etc |
    | Customer_Age | Customer age |
    | Account_Age | Account age in months |
    | Fraud | Target column (0 = Legitimate, 1 = Fraud) |
    """)
    st.markdown("""
    > 💡 **Tip:** Column names are auto-detected. Common aliases are supported.
    > If your column names are different, the app will attempt to map them automatically.
    """)


if __name__ == "__main__":
    main()
