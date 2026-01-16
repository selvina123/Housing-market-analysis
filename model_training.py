import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor
import lightgbm as lgb

# -----------------------------
# PATHS
# -----------------------------
data_path = "./Data/processed/updated_processed_data.csv"
model_path = "./Models/"
os.makedirs(model_path, exist_ok=True)

print("🔄 Loading processed dataset...")
data = pd.read_csv(data_path)

# -----------------------------
# CREATE ENGINEERED FEATURES
# -----------------------------
data["Housing_Market_Interaction"] = (
    data["City_Housing_Starts"] * data["market_heat_index"]
)

data["Housing_Sales_Ratio"] = (
    data["City_Housing_Starts"] / (data["sales_count_nowcast"] + 1)
)

# -----------------------------
# FEATURES + TARGET
# -----------------------------
features = [
    "City_Housing_Starts",
    "new_construction_sales_all_homes",
    "market_heat_index",
    "percent_sold_above_list_all_homes",
    "percent_sold_below_list_all_homes",
    "sales_count_nowcast",
    "total_transaction_value_all_homes",
    "zhvi_all_homes_smoothed",
    "Housing_Market_Interaction",
    "Housing_Sales_Ratio",
]

target = "zhvi_all_homes_smoothed"

X = data[features].copy()
y = data[target].copy()

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# SCALE FEATURES
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open(model_path + "scaler.pkl", "wb"))
print("📌 Saved scaler.pkl")


# -----------------------------
# MODELS TO TRAIN
# -----------------------------
models = {
    "random_forest.pkl": RandomForestRegressor(n_estimators=300, random_state=42),
    "gradient_boosting.pkl": GradientBoostingRegressor(random_state=42),
    "xgboost.pkl": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    ),
    "lightgbm.pkl": lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    ),
    "decision_tree.pkl": DecisionTreeRegressor(random_state=42),
    "linear_regression.pkl": LinearRegression(),
    "lasso_regression.pkl": Lasso(alpha=0.001)
}


# -----------------------------
# TRAIN & SAVE ALL MODELS
# -----------------------------
print("\n🚀 Starting model training...\n")

for filename, model in models.items():
    print(f"➡️ Training {filename}...")
    model.fit(X_train_scaled, y_train)
    pickle.dump(model, open(model_path + filename, "wb"))
    print(f"   ✔ Saved {filename}")

print("\n🎉 ALL MODELS TRAINED AND SAVED SUCCESSFULLY!\n")

from sklearn.metrics import mean_absolute_error, r2_score

print("\n🔍 Loading trained models for evaluation...")

# Load models you trained
xgb_model = pickle.load(open("./models/xgboost.pkl", "rb"))
rf_model = pickle.load(open("./models/random_forest.pkl", "rb"))
lgbm_model = pickle.load(open("./models/lightgbm.pkl", "rb"))
linear_model = pickle.load(open("./models/linear_regression.pkl", "rb"))
tree_model = pickle.load(open("./models/decision_tree.pkl", "rb"))

# Dictionary to store metrics
results = {}

def evaluate_model(name, model, X_test_scaled, y_test):
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    results[name] = {"MAE": mae, "R2": r2}
    print(f"\n📌 {name} Evaluation:")
    print(f"   MAE = {mae:,.4f}")
    print(f"   R²  = {r2:,.4f}")

# Run evaluation
evaluate_model("XGBoost", xgb_model, X_test_scaled, y_test)
evaluate_model("Random Forest", rf_model, X_test_scaled, y_test)
evaluate_model("LightGBM", lgbm_model, X_test_scaled, y_test)
evaluate_model("Linear Regression", linear_model, X_test_scaled, y_test)
evaluate_model("Decision Tree", tree_model, X_test_scaled, y_test)

# Save metrics
import json
with open("./outputs/model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n🎉 Evaluation complete! Metrics saved in outputs/model_metrics.json")

