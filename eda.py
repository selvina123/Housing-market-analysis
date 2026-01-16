import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Paths
data_path = "./data/processed/updated_processed_data.csv"

# Read the data
data = pd.read_csv(data_path)

# --------------------------------------------------------------
# ✅ FIX 1 — Convert 'Date' to datetime (VERY IMPORTANT)
# --------------------------------------------------------------
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Filter data to include only rows after 2010-01-01
data_timeseries = data[data["Date"] > "2010-01-01"]

# --------------------------------------------------------------
# 📈 1. Median Sale Price Over Time
# --------------------------------------------------------------
plt.figure(figsize=(16, 6))
sns.lineplot(
    data=data_timeseries, 
    x="Date", 
    y="median_sale_price_all_homes",
    linewidth=2,
    color="#1f77b4",
    ci=None
)

plt.title("Median Sale Price Over Time (Post-2010)", fontsize=16, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Median Sale Price", fontsize=12)

plt.xticks(rotation=45)
plt.locator_params(axis="x", nbins=12)  # show only 12 ticks (one per year)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 📊 2. Distribution of Market Heat Index
# --------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(data["market_heat_index"], kde=True, bins=30, color="#0096c7")
plt.title("Distribution of Market Heat Index", fontsize=14, fontweight="bold")
plt.xlabel("market_heat_index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 📊 3. Distribution of Percent Sold Above List
# --------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(data["percent_sold_above_list_all_homes"], kde=True, bins=30, color="#90be6d")
plt.title("Distribution of Percent Sold Above List (All Homes)", fontsize=14, fontweight="bold")
plt.xlabel("Percent Sold Above List")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 🔥 4. Correlation Heatmap (Selected Key Columns)
# --------------------------------------------------------------
selected_columns = [
    "median_sale_price_all_homes",
    "City_Housing_Starts",
    "market_heat_index",
    "percent_sold_above_list_all_homes",
    "percent_sold_below_list_all_homes",
    "sales_count_nowcast"
]

selected_data = data[selected_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(selected_data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap — Key Housing Variables", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
