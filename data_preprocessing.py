import pandas as pd
import os

# Paths
raw_data_path = "./data/raw/"
processed_data_path = "./data/processed/processed_data.csv"
city_housing_starts_path = "./data/raw/city_level_housing_starts.csv"
updated_data_path = "./data/processed/updated_processed_data.csv"

# File paths for all datasets
file_paths = {
    "new_construction_sales_all_homes": os.path.join(raw_data_path, "new_construction_sales_all_homes_monthly.csv"),
    "new_construction_sales_condo_coop": os.path.join(raw_data_path, "new_construction_sales_condo_coop_monthly.csv"),
    "new_construction_sales_sfr": os.path.join(raw_data_path, "new_construction_sales_sfr_monthly.csv"),
    "median_sale_price_all_homes": os.path.join(raw_data_path, "median_sale_price_all_homes_monthly.csv"),
    "median_list_price_all_homes": os.path.join(raw_data_path, "median_list_price_all_homes_monthly.csv"),
    "market_heat_index": os.path.join(raw_data_path, "market_heat_index_all_homes_monthly.csv"),
    "percent_sold_above_list_all_homes": os.path.join(raw_data_path, "percent_sold_above_list_all_homes_monthly.csv"),
    "percent_sold_below_list_all_homes": os.path.join(raw_data_path, "percent_sold_below_list_all_homes_monthly.csv"),
    "sales_count_nowcast": os.path.join(raw_data_path, "sales_count_nowcast_all_homes_monthly.csv"),
    "total_transaction_value_all_homes": os.path.join(raw_data_path, "total_transaction_value_all_homes_monthly.csv"),
    "zhvi_all_homes_smoothed": os.path.join(raw_data_path, "zhvi_all_homes_smoothed.csv"),
    "zhvi_condo_coop": os.path.join(raw_data_path, "zhvi_condo_coop.csv"),
    "zhvi_single_family_homes": os.path.join(raw_data_path, "zhvi_single_family_homes.csv"),
    "median_days_to_pending_all_homes": os.path.join(raw_data_path, "median_days_to_pending_all_homes_monthly.csv"),
    "median_days_to_close_all_homes": os.path.join(raw_data_path, "median_days_to_close_all_homes_monthly.csv"),
}

# Helper function to reshape datasets from wide to long format
def reshape_wide_to_long(df, id_vars, value_name):
    value_vars = [col for col in df.columns if col not in id_vars]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name="Date", value_name=value_name)
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
    return df_long

# Step 1: Load and reshape datasets
datasets = {}
for name, path in file_paths.items():
    if os.path.exists(path):
        print(f"Processing dataset: {name}")
        df = pd.read_csv(path)
        
        # Ensure all required keys are present
        required_keys = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
        for key in required_keys:
            if key not in df.columns:
                df[key] = None  # Add missing keys with default None
        
        reshaped_df = reshape_wide_to_long(df, required_keys, value_name=name)
        datasets[name] = reshaped_df
    else:
        print(f"File not found: {path}")

# Step 2: Merge all reshaped datasets
print("Merging datasets...")
merged_data = None
for name, df in datasets.items():
    if merged_data is None:
        merged_data = df
    else:
        merged_data = pd.merge(
            merged_data,
            df,
            on=["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", "Date"],
            how="outer"
        )

# Step 3: Load City Housing Starts and merge
print("Loading City Housing Starts data...")
city_housing_starts = pd.read_csv(city_housing_starts_path)

# Standardize RegionName and convert Date columns
print("Standardizing RegionName and Date formats...")
merged_data['RegionName'] = merged_data['RegionName'].str.strip().str.lower()
city_housing_starts['RegionName'] = city_housing_starts['RegionName'].str.strip().str.lower()
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
city_housing_starts['Date'] = pd.to_datetime(city_housing_starts['Date'])

# Merge City_Housing_Starts into merged_data
merged_data = pd.merge(
    merged_data,
    city_housing_starts[['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'Date', 'City_Housing_Starts']],
    on=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'Date'],
    how='left'
)

# Step 4: Add interaction features
print("Adding interaction features...")
merged_data["Housing_Market_Interaction"] = merged_data["City_Housing_Starts"] * merged_data["market_heat_index"]
merged_data["Housing_Sales_Ratio"] = merged_data["City_Housing_Starts"] / (merged_data["sales_count_nowcast"] + 1)

# Step 5: Handle missing values
print("Handling missing values...")
merged_data.fillna(method="ffill", inplace=True)
merged_data.fillna(method="bfill", inplace=True)
merged_data['City_Housing_Starts'].fillna(0, inplace=True)

# Step 6: Save the processed data
print(f"Saving updated processed data to {updated_data_path}...")
os.makedirs(os.path.dirname(updated_data_path), exist_ok=True)
merged_data.to_csv(updated_data_path, index=False)
print("Data preprocessing completed successfully.")
