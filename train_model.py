# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Trains multiple regression models on the California Housing dataset,
selects the best by cross-validated R², saves the full fitted pipeline
as best_model.pkl, and exports CV metrics to model_metrics.csv.

Run:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_CANDIDATES = ["Housing (1).csv", "housing.csv"]
data_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(
        "Could not find Housing (1).csv or housing.csv in the current directory."
    )
print(f"[INFO] Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"[INFO] Dataset shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Features / target split
# ---------------------------------------------------------------------------
TARGET = "median_house_value"
NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# 4. Candidate models
# ---------------------------------------------------------------------------
CANDIDATES = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}

# ---------------------------------------------------------------------------
# 5. Cross-validation
# ---------------------------------------------------------------------------
CV = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n[INFO] Running 5-fold cross-validation …")
for name, estimator in CANDIDATES.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    scores = cross_validate(
        pipeline, X_train, y_train, cv=CV,
        scoring=["r2", "neg_root_mean_squared_error"],
        n_jobs=-1,
    )
    mean_r2 = scores["test_r2"].mean()
    mean_rmse = -scores["test_neg_root_mean_squared_error"].mean()
    results.append({"model": name, "mean_r2": mean_r2, "mean_rmse": mean_rmse})
    print(f"  {name:<26} R²={mean_r2:.4f}  RMSE={mean_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 6. Save metrics CSV
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("model_metrics.csv", index=False)
print("\n[INFO] Saved model_metrics.csv")

# ---------------------------------------------------------------------------
# 7. Select best model
# ---------------------------------------------------------------------------
best_row = metrics_df.loc[metrics_df["mean_r2"].idxmax()]
best_name = best_row["model"]
print(f"\n[INFO] Best model: {best_name}  (R²={best_row['mean_r2']:.4f})")

# ---------------------------------------------------------------------------
# 8. Refit best pipeline on full training data
# ---------------------------------------------------------------------------
best_estimator = CANDIDATES[best_name]
best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_estimator)])
best_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 9. Evaluate on held-out test set
# ---------------------------------------------------------------------------
y_pred = best_pipeline.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"[INFO] Test set — R²={test_r2:.4f}  RMSE={test_rmse:,.0f}")

# ---------------------------------------------------------------------------
# 10. Save pipeline
# ---------------------------------------------------------------------------
joblib.dump(best_pipeline, "best_model.pkl")
print("[INFO] Saved best_model.pkl")

# ---------------------------------------------------------------------------
# 11. Print summary table
# ---------------------------------------------------------------------------
print("\n=== Model Comparison ===")
print(metrics_df.to_string(index=False))
print(f"\nSelected: {best_name}")
