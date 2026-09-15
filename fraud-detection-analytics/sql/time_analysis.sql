-- ============================================================
-- Time Analysis Queries
-- Fraud Detection & Transaction Analytics Dashboard
-- ============================================================

-- 1. Fraud by Hour of Day
SELECT
    CAST(STRFTIME('%H', Transaction_Time) AS INTEGER) AS hour,
    COUNT(*)                                          AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)       AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                          AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;

-- 2. Fraud by Day of Week
SELECT
    CASE CAST(STRFTIME('%w', Transaction_Date) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_of_week,
    CAST(STRFTIME('%w', Transaction_Date) AS INTEGER) AS day_num,
    COUNT(*)                                          AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)       AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                          AS fraud_rate_pct
FROM transactions
GROUP BY day_of_week, day_num
ORDER BY day_num;

-- 3. Monthly Fraud Trend
SELECT
    STRFTIME('%Y-%m', Transaction_Date) AS month,
    COUNT(*)                            AS total,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)                    AS fraud_count,
    ROUND(SUM(CASE WHEN Fraud = 1 THEN Amount ELSE 0 END), 2)     AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                                        AS fraud_rate_pct
FROM transactions
GROUP BY month
ORDER BY month;

-- 4. Peak Fraud Hours (Top 5)
SELECT
    CAST(STRFTIME('%H', Transaction_Time) AS INTEGER) AS hour,
    SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)       AS fraud_count,
    ROUND(100.0 * SUM(CASE WHEN Fraud = 1 THEN 1 ELSE 0 END)
              / COUNT(*), 2)                          AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY fraud_count DESC
LIMIT 5;

-- 5. Transaction Volume by Month
SELECT
    STRFTIME('%Y-%m', Transaction_Date) AS month,
    COUNT(*)                            AS total_transactions,
    ROUND(SUM(Amount), 2)               AS total_amount,
    ROUND(AVG(Amount), 2)               AS avg_amount
FROM transactions
GROUP BY month
ORDER BY month;
