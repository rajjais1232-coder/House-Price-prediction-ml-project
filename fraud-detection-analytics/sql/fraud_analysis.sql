-- ============================================================
-- Fraud Analysis Queries
-- Fraud Detection & Transaction Analytics Dashboard
-- ============================================================

-- 1. Fraud Rate by Hour of Day
SELECT
    CAST(STRFTIME('%H', Transaction_Time) AS INTEGER)  AS hour_of_day,
    COUNT(*)                                           AS total_transactions,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)        AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                           AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 2. Fraud by Payment Method
SELECT
    Payment_Method,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)        AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                           AS fraud_rate_pct
FROM transactions
GROUP BY Payment_Method
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud by Device Type
SELECT
    Device_Type,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)        AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                           AS fraud_rate_pct
FROM transactions
GROUP BY Device_Type
ORDER BY fraud_rate_pct DESC;

-- 4. Fraud by Transaction Type
SELECT
    Transaction_Type,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)        AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                           AS fraud_rate_pct
FROM transactions
GROUP BY Transaction_Type
ORDER BY fraud_rate_pct DESC;

-- 5. International vs Domestic Fraud
SELECT
    International_Transaction,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)        AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                           AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY International_Transaction;

-- 6. Amount Distribution for Fraud vs Legitimate
SELECT
    CASE WHEN Fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS status,
    ROUND(MIN(Amount), 2)   AS min_amount,
    ROUND(MAX(Amount), 2)   AS max_amount,
    ROUND(AVG(Amount), 2)   AS avg_amount,
    COUNT(*)                AS count
FROM transactions
GROUP BY Fraud;

-- 7. High-Value Fraud Transactions (Top 20)
SELECT
    Transaction_ID, Customer_ID, Transaction_Date, Amount,
    Merchant, Location, Payment_Method, Device_Type
FROM transactions
WHERE Fraud = 1
ORDER BY Amount DESC
LIMIT 20;

-- 8. Fraud Trend Over Time (weekly)
SELECT
    STRFTIME('%Y-W%W', Transaction_Date) AS week,
    COUNT(*)                             AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)   AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY week
ORDER BY week;
