-- ============================================================
-- Merchant Analysis Queries
-- Fraud Detection & Transaction Analytics Dashboard
-- ============================================================

-- 1. Merchant Summary
SELECT
    Merchant,
    Merchant_Category,
    COUNT(*)                                                    AS total_transactions,
    ROUND(SUM(Amount), 2)                                      AS total_amount,
    ROUND(AVG(Amount), 2)                                      AS avg_amount,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY Merchant, Merchant_Category
ORDER BY fraud_amount DESC;

-- 2. Top Merchants by Fraud Count
SELECT
    Merchant,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)  AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY Merchant
ORDER BY fraud_count DESC
LIMIT 10;

-- 3. Top Merchants by Fraud Amount
SELECT
    Merchant,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)  AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY Merchant
ORDER BY fraud_amount DESC
LIMIT 10;

-- 4. Fraud by Merchant Category
SELECT
    Merchant_Category,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY Merchant_Category
ORDER BY fraud_rate_pct DESC;

-- 5. High-Risk Merchants (fraud rate > 20% with at least 10 transactions)
SELECT
    Merchant,
    Merchant_Category,
    COUNT(*)                                                    AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                    AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2) AS fraud_exposure
FROM transactions
GROUP BY Merchant
HAVING fraud_rate_pct > 20 AND total >= 10
ORDER BY fraud_exposure DESC;
