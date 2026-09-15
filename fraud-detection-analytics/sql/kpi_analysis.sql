-- ============================================================
-- KPI Analysis Queries
-- Fraud Detection & Transaction Analytics Dashboard
-- ============================================================

-- 1. Core Transaction KPIs
SELECT
    COUNT(*)                                            AS total_transactions,
    ROUND(SUM(Amount), 2)                               AS total_amount,
    ROUND(AVG(Amount), 2)                               AS avg_amount,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END), 0)          AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                            AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2)      AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END)
              / NULLIF(SUM(Amount), 0), 2)              AS fraud_amount_pct,
    COUNT(DISTINCT Customer_ID)                         AS unique_customers,
    COUNT(DISTINCT Merchant)                            AS unique_merchants
FROM transactions;

-- 2. Average and Median Transaction Amount
SELECT
    ROUND(AVG(Amount), 2)   AS avg_transaction_amount,
    -- SQLite has no MEDIAN; approximate with ordering
    Amount                  AS median_amount
FROM transactions
ORDER BY Amount
LIMIT 1
OFFSET (SELECT COUNT(*) FROM transactions) / 2;

-- 3. Fraud vs Legitimate Split
SELECT
    CASE WHEN Fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS transaction_status,
    COUNT(*)                    AS count,
    ROUND(SUM(Amount), 2)       AS total_amount,
    ROUND(AVG(Amount), 2)       AS avg_amount,
    ROUND(100.0 * COUNT() / (SELECT COUNT(*) FROM transactions), 2) AS pct_of_total
FROM transactions
GROUP BY Fraud;

-- 4. Daily KPI Summary
SELECT
    Transaction_Date,
    COUNT(*)                                                        AS daily_transactions,
    ROUND(SUM(Amount), 2)                                          AS daily_amount,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                    AS daily_fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                        AS daily_fraud_rate
FROM transactions
GROUP BY Transaction_Date
ORDER BY Transaction_Date;

-- 5. Monthly KPI Summary
SELECT
    STRFTIME('%Y-%m', Transaction_Date) AS month,
    COUNT(*)                            AS monthly_transactions,
    ROUND(SUM(Amount), 2)               AS monthly_amount,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END) AS monthly_fraud,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)            AS monthly_fraud_rate
FROM transactions
GROUP BY month
ORDER BY month;
