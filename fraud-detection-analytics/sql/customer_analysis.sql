-- ============================================================
-- Customer Analysis Queries
-- Fraud Detection & Transaction Analytics Dashboard
-- ============================================================

-- 1. Customer-Level Summary
SELECT
    Customer_ID,
    COUNT(*)                                                    AS total_transactions,
    ROUND(SUM(Amount), 2)                                      AS total_amount,
    ROUND(AVG(Amount), 2)                                      AS avg_amount,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount,
    MAX(Failed_Transactions)                                   AS max_failed_txns
FROM transactions
GROUP BY Customer_ID
ORDER BY fraud_count DESC;

-- 2. High-Risk Customers (fraud rate > 30% AND at least 3 transactions)
SELECT
    Customer_ID,
    COUNT(*)                                                    AS total_transactions,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_exposure
FROM transactions
GROUP BY Customer_ID
HAVING fraud_rate_pct > 30 AND total_transactions >= 3
ORDER BY fraud_exposure DESC;

-- 3. Customer Age Segment Analysis
SELECT
    CASE
        WHEN Customer_Age < 25 THEN '18-24'
        WHEN Customer_Age < 35 THEN '25-34'
        WHEN Customer_Age < 45 THEN '35-44'
        WHEN Customer_Age < 55 THEN '45-54'
        WHEN Customer_Age < 65 THEN '55-64'
        ELSE '65+'
    END AS age_group,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct
FROM transactions
GROUP BY age_group
ORDER BY age_group;

-- 4. Account Age vs Fraud Rate
SELECT
    CASE
        WHEN Account_Age < 6   THEN '0-6 months'
        WHEN Account_Age < 12  THEN '6-12 months'
        WHEN Account_Age < 24  THEN '1-2 years'
        WHEN Account_Age < 60  THEN '2-5 years'
        ELSE '5+ years'
    END AS account_tenure,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct
FROM transactions
GROUP BY account_tenure;

-- 5. Customers with High Failed Transactions
SELECT
    Customer_ID,
    MAX(Failed_Transactions)   AS max_failed,
    COUNT(*)                   AS total_transactions,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END) AS fraud_count
FROM transactions
GROUP BY Customer_ID
HAVING max_failed > 5
ORDER BY max_failed DESC
LIMIT 50;
