# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏠 Housing Market Future Value Prediction (ZHVI)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white;'>Predict home value changes using ML models.<br>Target = <b>zhvi_all_homes_smoothed</b></p>", unsafe_allow_html=True)

# Paths
data_path = "./Data/processed/updated_processed_data.csv"
model_path = "./Models/"
output_path = "./outputs/"

# Load Data + Scaler
@st.cache_data
def load_data():
    data = pd.read_csv(data_path)
    with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return data, scaler

data, scaler = load_data()

# Model List
model_files = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Linear Regression": "linear_regression.pkl",
    "Lasso Regression": "lasso_regression.pkl",
}

def load_model(name):
    with open(os.path.join(model_path, name), "rb") as f:
        return pickle.load(f)

# Inputs UI
st.subheader("🔍 Prediction Settings")

col1, col2, col3 = st.columns(3)
with col1:
    selected_model = st.selectbox("Choose Model", list(model_files.keys()))
with col2:
    selected_city = st.selectbox("Choose City", sorted(data["RegionName"].unique()))
with col3:
    extra_units = st.number_input("Extra Housing Units", min_value=0, step=1)

model = load_model(model_files[selected_model])
city_data = data[data["RegionName"] == selected_city]

# Ensure interaction features
city_data["Housing_Market_Interaction"] = (
    city_data["City_Housing_Starts"] * city_data["market_heat_index"]
)
city_data["Housing_Sales_Ratio"] = (
    city_data["City_Housing_Starts"] / (city_data["sales_count_nowcast"] + 1)
)

# Prediction Function
def predict(df, extra):
    df = df.copy()
    df["City_Housing_Starts"] += extra
    df["Housing_Market_Interaction"] = (
        df["City_Housing_Starts"] * df["market_heat_index"]
    )
    df["Housing_Sales_Ratio"] = (
        df["City_Housing_Starts"] / (df["sales_count_nowcast"] + 1)
    )

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

    X = df[features]
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)

# Predictions
base = predict(city_data, 0)
new = predict(city_data, extra_units)

st.markdown(f"<h3 style='color:white;'>📊 Predictions for <b>{selected_city}</b></h3>", unsafe_allow_html=True)

# Results Card
st.markdown(
    f"""
    <div class="pred-card">
        <h4>Baseline ZHVI: ${np.mean(base):,.2f}</h4>
        <h4>After Adding {extra_units} Units: ${np.mean(new):,.2f}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(x=city_data["Date"], y=base, mode='lines', name="Baseline", line=dict(color="#00c0ff")))
fig.add_trace(go.Scatter(x=city_data["Date"], y=new, mode='lines', name="With Extra Units", line=dict(color="#ff007f")))

fig.update_layout(
    title=f"Impact of Extra Housing Units in {selected_city}",
    xaxis_title="Date",
    yaxis_title="Predicted ZHVI",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white")
)

st.plotly_chart(fig, use_container_width=True)

