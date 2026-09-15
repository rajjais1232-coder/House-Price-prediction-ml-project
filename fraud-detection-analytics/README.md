# 🔍 Fraud Detection & Transaction Analytics Dashboard

> **A production-style end-to-end analytics project for Data Analyst portfolios.**
> Combines SQL analytics, exploratory data analysis, business KPIs, interactive visualisations, and Machine Learning fraud prediction in a single Streamlit dashboard.

---

## 📌 Project Overview

Financial fraud costs businesses and consumers billions of dollars annually. This project builds a complete **Fraud Detection & Transaction Analytics Dashboard** that helps analysts and investigators:

- Understand fraud patterns across time, location, merchant, and payment method
- Identify high-risk customers and merchants
- Quantify the financial impact of fraud
- Use Machine Learning to flag suspicious transactions in real time

The application is built with **Python, Pandas, SQL/SQLite, Scikit-learn, Plotly, and Streamlit**, and is deployable to **Streamlit Community Cloud** from a GitHub repository.

---

## 🎯 Business Problem

A financial services company processes thousands of transactions daily. The fraud team needs answers to:

- What percentage of transactions are fraudulent?
- Which locations, merchants, and payment methods are highest risk?
- Which customers are most associated with fraud?
- When does fraud occur most frequently?
- How much money is lost to fraud?
- Which transactions should investigators prioritise?

---

## 🏗️ Objectives

1. **Ingest & Clean** raw transaction CSV data with automated column detection
2. **Analyse** transactions using SQL queries and Python/Pandas analytics
3. **Calculate KPIs** including fraud rate, fraud amount, average transaction value
4. **Visualise** trends and patterns across time, geography, merchants, and customers
5. **Build ML Models** to predict fraud probability using Logistic Regression, Decision Tree, and Random Forest
6. **Deploy** an interactive dashboard suitable for business stakeholders

---

## 📁 Dataset Description

The application works with any CSV transaction dataset containing these fields:

| Column | Type | Description |
|---|---|---|
| Transaction_ID | String | Unique transaction identifier |
| Customer_ID | String | Customer identifier |
| Transaction_Date | Date | Date of transaction (YYYY-MM-DD) |
| Transaction_Time | Time | Time of transaction (HH:MM:SS) |
| Amount | Float | Transaction amount in USD |
| Merchant | String | Merchant name |
| Merchant_Category | String | Merchant business category |
| Location | String | Transaction location |
| Transaction_Type | String | Purchase / Transfer / Withdrawal / Refund / Subscription |
| Payment_Method | String | Credit Card / Debit Card / PayPal / etc |
| Device_Type | String | Mobile / Desktop / Tablet / POS Terminal / ATM |
| Customer_Age | Integer | Customer age in years |
| Account_Age | Integer | Account age in months |
| Previous_Transactions | Integer | Count of previous transactions |
| Failed_Transactions | Integer | Count of recent failed transactions |
| Transaction_Frequency | Integer | Transactions per week |
| International_Transaction | String | Yes / No |
| Fraud | Integer | Target label: 0 = Legitimate, 1 = Fraudulent |

> **Column names are auto-detected.** Common aliases (e.g. `is_fraud`, `trans_date`, `amt`) are supported automatically.

The included sample dataset (`data/sample_transactions.csv`) contains **10,000 synthetic transactions** with a ~13.5% fraud rate.

---

## 🧹 Data Cleaning

The following steps are applied automatically after upload:

1. **Duplicate removal** — exact duplicate rows are dropped
2. **Date parsing** — Transaction_Date is parsed to datetime
3. **Amount validation** — non-numeric values coerced; negative amounts removed
4. **Fraud label standardisation** — Yes/True/1 mapped to 1; all others to 0
5. **Missing value imputation** — numeric columns filled with median; categorical with mode
6. **International_Transaction normalisation** — standardised to Yes/No
7. **Feature derivation** — Transaction_Month, DayOfWeek, Hour extracted from date/time

> The original uploaded dataset is never modified.

---

## 🔍 Exploratory Data Analysis

### Transaction KPIs
| KPI | Formula |
|---|---|
| Total Transactions | COUNT(*) |
| Total Amount | SUM(Amount) |
| Fraud Rate | SUM(Fraud) / COUNT(*) × 100 |
| Fraud Amount | SUM(Amount WHERE Fraud=1) |
| Fraud Amount % | Fraud Amount / Total Amount × 100 |
| Avg Fraud Amount | AVG(Amount WHERE Fraud=1) |

### Analysis Dimensions
- **Time** — Fraud by hour, day, day-of-week, month, trend over time
- **Customer** — Per-customer fraud rate, high-risk customer identification, age/tenure segments
- **Merchant** — Fraud by merchant, category, top fraud merchants by count and amount
- **Geography** — Fraud rate by location, high-risk locations
- **Payment** — Fraud by payment method, device type, transaction type, international vs domestic

---

## 🗄️ SQL Analytics

SQL queries are organised in the `sql/` folder:

| File | Description |
|---|---|
| `kpi_analysis.sql` | Core KPIs: total transactions, amounts, fraud rates, daily/monthly summaries |
| `fraud_analysis.sql` | Fraud by hour, payment method, device, type, amount distribution, top fraud transactions |
| `customer_analysis.sql` | Customer summaries, high-risk customers, age/tenure segment analysis |
| `merchant_analysis.sql` | Merchant fraud summary, top fraud merchants, fraud by category |
| `time_analysis.sql` | Fraud by hour, day of week, weekly trends, peak fraud hours |

Example query — **High-Risk Merchants**:
```sql
SELECT Merchant, Merchant_Category,
       COUNT(*) AS total,
       SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
       ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY Merchant
HAVING fraud_rate_pct > 20 AND total >= 10
ORDER BY fraud_rate_pct DESC;
```

---

## 📊 Dashboard Features

### Sidebar
- CSV file upload
- Sample dataset loader
- Date range filter
- Merchant category, location, payment method, transaction type filters
- Fraud status filter (All / Fraud Only / Legitimate Only)

### Page 1 — Executive Overview
KPI cards + transaction volume over time + fraud trend + fraud vs legitimate pie + fraud by payment method + fraud by merchant category + fraud by location heatmap

### Page 2 — Fraud Analysis
Fraud rate by hour + by transaction type + by device (pie) + amount distribution overlay + international vs domestic comparison + top fraud merchants + monthly fraud area chart + top fraud transactions table

### Page 3 — Customer Risk Analytics
Customer KPIs + fraud by age group + fraud by account tenure + customer scatter (amount vs fraud rate) + high-risk customer table + failed transactions analysis

### Page 4 — Merchant Analytics
Transaction volume by merchant + fraud count by merchant + fraud amount by merchant + fraud rate by category + full merchant summary table

### Page 5 — ML Fraud Prediction
- Model training (Logistic Regression, Decision Tree, Random Forest)
- Model comparison table (Precision, Recall, F1, ROC-AUC)
- Confusion matrix heatmap
- ROC and Precision-Recall curves
- Feature importance chart
- Anomaly detection (Isolation Forest)
- Interactive transaction prediction form with risk gauge and risk factor explanation

### Download Buttons
- Filtered transactions CSV
- Fraudulent transactions CSV
- High-risk customers CSV
- Model comparison CSV
- Anomalies CSV

---

## 🤖 Machine Learning Approach

### Models Trained
| Model | Notes |
|---|---|
| Logistic Regression | Baseline; class_weight='balanced' |
| Decision Tree | max_depth=8; class_weight='balanced' |
| Random Forest | 100 estimators; class_weight='balanced' |

### Class Imbalance Handling
Fraud datasets are inherently imbalanced (typically 1–20% fraud). All models use `class_weight='balanced'` to prevent the model from ignoring the minority fraud class.

### Evaluation Metrics
**Accuracy is NOT used as the primary metric** (it is misleading on imbalanced datasets).

| Metric | Why it matters |
|---|---|
| **Precision** | Of predicted frauds, how many were real? (minimises false alarms) |
| **Recall** | Of actual frauds, how many did we catch? (minimises missed fraud) |
| **F1 Score** | Harmonic mean of Precision and Recall — primary selection criterion |
| **ROC-AUC** | Overall model discrimination ability |
| **Average Precision** | Area under Precision-Recall curve |

### Best Model Selection
The best model is selected by **highest F1 Score**, giving balanced weight to both catching fraud (Recall) and avoiding false positives (Precision).

---

## 🎯 Fraud Detection Methodology

### Risk Categories
| Probability | Risk Level |
|---|---|
| 0% – 30% | 🟢 Low Risk |
| 30% – 70% | 🟡 Medium Risk |
| 70% – 100% | 🔴 High Risk |

### Risk Factors Considered
- International transaction (higher risk)
- Transaction amount > $2,000 (higher risk)
- High failed transaction count (>5)
- High-risk payment methods (Crypto, Gift Card)
- High-risk merchant categories (Money Transfer, ATM, Unknown)
- New account (account age < 6 months)
- High-risk transaction types (Transfer, Withdrawal)

### Anomaly Detection
Isolation Forest is used for unsupervised anomaly detection — useful when labelled fraud data is unavailable. It identifies unusual transactions based on Amount, Failed Transactions, Account Age, and Transaction Frequency.

---

## 💡 Business Insights

Based on typical fraud patterns detected by this model:

1. **International transactions** show significantly higher fraud rates (2–3×)
2. **Crypto and Gift Card** payments are disproportionately associated with fraud
3. **New accounts** (< 6 months) carry higher fraud risk
4. **Transfers and Withdrawals** have higher fraud rates than Purchases
5. **High failed transaction counts** are a strong fraud signal
6. **Late night / early morning** hours (0:00–5:00) often show elevated fraud rates
7. **ATM and Money Transfer** merchant categories show the highest fraud exposure

---

## 📋 Business Recommendations

1. **Flag high-risk international transactions** for manual review before processing
2. **Implement real-time scoring** using the trained model for transactions above $2,000
3. **Increase friction** (2FA, verification) for Crypto and Gift Card payment methods
4. **Monitor new accounts** (< 6 months) with lower transaction limits
5. **Investigate high-failed-transaction customers** as a leading indicator
6. **Set up automated alerts** for customers matching the high-risk profile
7. **Prioritise merchants** in Money Transfer and Unknown categories for audits
8. **Focus investigator capacity** on the high-risk customers table

---

## 🏛️ Project Architecture

```
fraud-detection-analytics/
│
├── data/
│   ├── sample_transactions.csv     # 10,000 synthetic transactions
│   └── generate_sample.py          # Script to regenerate sample data
│
├── sql/
│   ├── kpi_analysis.sql
│   ├── fraud_analysis.sql
│   ├── customer_analysis.sql
│   ├── merchant_analysis.sql
│   └── time_analysis.sql
│
├── models/
│   └── fraud_model.pkl             # Saved best model + preprocessing artefacts
│
├── notebooks/
│   └── fraud_eda.ipynb             # Exploratory analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Ingestion, cleaning, quality reports
│   ├── analytics.py                # KPIs, aggregations, business analytics
│   ├── feature_engineering.py      # Feature matrix, encoding, scaling
│   ├── model_training.py           # Model training, comparison, anomaly detection
│   └── model_evaluation.py         # Metrics, charts, risk categorisation
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data manipulation and analytics |
| NumPy | Numerical operations |
| Scikit-learn | ML models, preprocessing, evaluation |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts and visualisations |
| Joblib | Model serialisation |
| SQLite | SQL analytics queries |
| Matplotlib / Seaborn | Static visualisations (notebooks) |

---

## ⚙️ Installation Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-analytics.git
cd fraud-detection-analytics
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**Quick start:**
1. Click **"Use Sample Dataset"** in the sidebar
2. Explore the **Executive Overview** page
3. Navigate to **ML Fraud Prediction** and click **Train Models**
4. Use the prediction form to score a custom transaction

**Upload your own data:**
1. Click **"Upload CSV Dataset"** in the sidebar
2. Select your transaction CSV file
3. The app auto-detects columns and cleans the data
4. All dashboard pages update automatically

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set **Main file path** to `app.py`
5. Click **Deploy**

---

## 🔮 Future Improvements

- [ ] Real-time transaction scoring via REST API (FastAPI)
- [ ] SHAP explainability for individual predictions
- [ ] Graph-based fraud network analysis (customer–merchant–IP relationships)
- [ ] Time-series anomaly detection (ARIMA/LSTM)
- [ ] Automated report generation (PDF export)
- [ ] PostgreSQL / cloud database integration
- [ ] Role-based access control for multi-user deployment
- [ ] Email alert integration for high-risk transaction triggers
- [ ] Model retraining pipeline with new labelled data
- [ ] A/B testing framework for model comparison in production

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with Python · Pandas · Scikit-learn · Streamlit · Plotly*
