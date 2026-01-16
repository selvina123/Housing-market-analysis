import pandas as pd
import shap
import pickle
import os
import matplotlib.pyplot as plt

# ================================
# Paths
# ================================
data_path = "./data/processed/updated_processed_data.csv"
model_path = "./models/"
output_path = "./outputs/"
os.makedirs(output_path, exist_ok=True)

# ================================
# Load Data
# ================================
print("Loading processed dataset...")
data = pd.read_csv(data_path)

# Add interaction features if missing
if "Housing_Market_Interaction" not in data.columns or "Housing_Sales_Ratio" not in data.columns:
    print("Adding interaction features...")
    data["Housing_Market_Interaction"] = data["City_Housing_Starts"] * data["market_heat_index"]
    data["Housing_Sales_Ratio"] = data["City_Housing_Starts"] / (data["sales_count_nowcast"] + 1)

# ================================
# Define Features + Target
# ================================
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

# IMPORTANT:
# Your final training used ZHVI as the target
target = "zhvi_all_homes_smoothed"

X = data[features].fillna(0)
y = data[target].fillna(0)

# ================================
# Load Model + Scaler
# ================================
print("Loading best-trained model for SHAP...")

# FIXED: use real file present in your folder
model_file = os.path.join(model_path, "xgboost.pkl")
scaler_file = os.path.join(model_path, "scaler.pkl")

with open(model_file, "rb") as f:
    model = pickle.load(f)

with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# Scale features
X_scaled = scaler.transform(X)

# ================================
# SHAP Explainer
# ================================
print("Initializing SHAP Tree Explainer...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_scaled)

# ================================
# SHAP Summary Plot
# ================================
print("Generating SHAP summary plot...")

plt.figure()
shap.summary_plot(shap_values, X, feature_names=features, show=False)

summary_plot_path = os.path.join(output_path, "shap_summary_plot.png")
plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"🔥 SHAP summary plot saved to: {summary_plot_path}")
