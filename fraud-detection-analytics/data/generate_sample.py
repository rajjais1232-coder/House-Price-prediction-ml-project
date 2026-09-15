"""
Script to generate a realistic sample transactions dataset for fraud detection.
Run once: python data/generate_sample.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 10000

merchants = [
    "Amazon", "Walmart", "Target", "Best Buy", "Starbucks",
    "McDonald's", "Shell Gas", "CVS Pharmacy", "Home Depot", "Netflix",
    "Uber", "Lyft", "Airbnb", "Apple Store", "Google Play",
    "PayPal", "Venmo Transfer", "Chase ATM", "Bank Transfer", "Unknown Merchant"
]
merchant_categories = {
    "Amazon": "E-commerce", "Walmart": "Retail", "Target": "Retail",
    "Best Buy": "Electronics", "Starbucks": "Food & Beverage",
    "McDonald's": "Food & Beverage", "Shell Gas": "Gas & Fuel",
    "CVS Pharmacy": "Healthcare", "Home Depot": "Home Improvement",
    "Netflix": "Subscription", "Uber": "Transportation", "Lyft": "Transportation",
    "Airbnb": "Travel", "Apple Store": "Electronics", "Google Play": "Subscription",
    "PayPal": "Money Transfer", "Venmo Transfer": "Money Transfer",
    "Chase ATM": "ATM", "Bank Transfer": "Money Transfer", "Unknown Merchant": "Unknown"
}
locations = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
    "Dallas, TX", "San Jose, CA", "Miami, FL", "Seattle, WA",
    "Denver, CO", "Boston, MA", "Atlanta, GA", "Las Vegas, NV",
    "International - UK", "International - Canada", "International - Mexico", "International - Unknown"
]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Crypto", "Gift Card"]
device_types = ["Mobile", "Desktop", "Tablet", "POS Terminal", "ATM"]
transaction_types = ["Purchase", "Transfer", "Withdrawal", "Refund", "Subscription"]

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 6, 30)

def random_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

transaction_ids = [f"TXN{str(i).zfill(8)}" for i in range(1, N + 1)]
customer_ids = [f"CUST{str(random.randint(1, 1500)).zfill(5)}" for _ in range(N)]
transaction_dates = [random_date(start_date, end_date) for _ in range(N)]

merchant_list = random.choices(merchants, k=N)
merchant_categories_list = [merchant_categories[m] for m in merchant_list]
location_list = random.choices(locations, weights=[8,7,6,6,5,5,5,5,5,5,4,4,4,4,4,3,2,2,2,1], k=N)
payment_list = random.choices(payment_methods, weights=[35,30,15,10,5,5], k=N)
device_list = random.choices(device_types, weights=[40,25,10,20,5], k=N)
type_list = random.choices(transaction_types, weights=[50,20,10,10,10], k=N)

customer_ages = np.random.randint(18, 80, size=N)
account_ages = np.random.randint(1, 180, size=N)
prev_transactions = np.random.randint(0, 500, size=N)
failed_transactions = np.random.randint(0, 20, size=N)
transaction_freq = np.random.randint(1, 50, size=N)
international = ["Yes" if "International" in loc else "No" for loc in location_list]

amounts = np.where(
    np.array(type_list) == "Withdrawal",
    np.random.lognormal(mean=5.5, sigma=0.8, size=N),
    np.random.lognormal(mean=4.8, sigma=1.2, size=N)
)
amounts = np.round(np.clip(amounts, 1, 15000), 2)

# Fraud logic — biased toward suspicious patterns
fraud_prob = np.zeros(N)
fraud_prob += np.where(np.array(international) == "Yes", 0.12, 0.0)
fraud_prob += np.where(amounts > 2000, 0.10, 0.0)
fraud_prob += np.where(np.array(failed_transactions) > 5, 0.08, 0.0)
fraud_prob += np.where(np.array(payment_list).isin(["Crypto", "Gift Card"]) if False else
                        np.isin(payment_list, ["Crypto", "Gift Card"]), 0.12, 0.0)
fraud_prob += np.where(np.array(merchant_list) == "Unknown Merchant", 0.15, 0.0)
fraud_prob += np.where(np.array(device_list) == "ATM", 0.05, 0.0)
fraud_prob += np.where(np.array(account_ages) < 10, 0.06, 0.0)
fraud_prob += np.where(np.array(type_list).isin(["Transfer"]) if False else
                        np.isin(type_list, ["Transfer", "Withdrawal"]), 0.05, 0.0)
fraud_prob += 0.03  # base fraud rate
fraud_prob = np.clip(fraud_prob, 0, 0.75)
fraud = np.random.binomial(1, fraud_prob)

df = pd.DataFrame({
    "Transaction_ID": transaction_ids,
    "Customer_ID": customer_ids,
    "Transaction_Date": [d.strftime("%Y-%m-%d") for d in transaction_dates],
    "Transaction_Time": [d.strftime("%H:%M:%S") for d in transaction_dates],
    "Amount": amounts,
    "Merchant": merchant_list,
    "Merchant_Category": merchant_categories_list,
    "Location": location_list,
    "Transaction_Type": type_list,
    "Payment_Method": payment_list,
    "Device_Type": device_list,
    "Customer_Age": customer_ages,
    "Account_Age": account_ages,
    "Previous_Transactions": prev_transactions,
    "Failed_Transactions": failed_transactions,
    "Transaction_Frequency": transaction_freq,
    "International_Transaction": international,
    "Fraud": fraud
})

df.to_csv("data/sample_transactions.csv", index=False)
print(f"Generated {N} transactions | Fraud rate: {fraud.mean():.2%}")
print(f"Saved to data/sample_transactions.csv")
