import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from datetime import datetime

# Load the dataset
csv_file = "data/all_cities.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["date"])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

# Add MLflow setup
mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
mlflow.set_experiment("weather")

# Data Cleaning

# 1. Handle missing values
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
non_numeric_columns = df.select_dtypes(exclude=["float64", "int64"]).columns
print(f"Numeric columns: {numeric_columns}")
print(f"Non-numeric columns: {non_numeric_columns}")

# Fill missing values for numeric columns with mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values for non-numeric columns with mode (most frequent value)
for col in non_numeric_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2. Correct data types
# Ensure 'date' is datetime and other numerical columns are in correct format
df["date"] = pd.to_datetime(df["date"])
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Remove duplicates
initial_row_count = df.shape[0]
df.drop_duplicates(subset=["date", "city"], inplace=True)
final_row_count = df.shape[0]
print(f"Removed {initial_row_count - final_row_count} duplicate rows.")

# Feature Engineering

# Convert date to numerical format
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

# Create cyclical features for month and day of year
df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

# Example: Create a 'month' column from 'date'
df["month"] = df["date"].dt.month

# Exploratory Data Analysis (EDA)

# Statistical summaries
print(df.describe())


# Data Transformation

# Example: Normalize numerical features
from sklearn.preprocessing import MinMaxScaler

# Scale features
feature_columns = [
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]

# Define target columns
target_columns = ["temperature_2m", "precipitation"]

# Create the scalers directory if it doesn't exist
scalers_dir = "data/scalers/"
os.makedirs(scalers_dir, exist_ok=True)

# Save separate scalers for features and targets
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale features and targets separately
df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
df[target_columns] = target_scaler.fit_transform(df[target_columns])

# Save both scalers
feature_scaler_filename = os.path.join(scalers_dir, "feature_scaler.joblib")
target_scaler_filename = os.path.join(scalers_dir, "target_scaler.joblib")
joblib.dump(feature_scaler, feature_scaler_filename)
joblib.dump(target_scaler, target_scaler_filename)
print(f"Feature scaler saved to {feature_scaler_filename}")
print(f"Target scaler saved to {target_scaler_filename}")

# Save the processed data
processed_csv_file = "data/all_cities_processed.csv"
df.to_csv(processed_csv_file, index=False)
print(f"Processed data saved to {processed_csv_file}.")

run_name = f"data-processing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Log basic parameters
    mlflow.log_params(
        {
            "input_file": csv_file,
            "initial_rows": initial_row_count,
            "initial_columns": len(df.columns),
            "data_start_date": df["date"].min().strftime("%Y-%m-%d"),
            "data_end_date": df["date"].max().strftime("%Y-%m-%d"),
            "num_cities": len(df["city"].unique()),
        }
    )

    # Create temporary CSV files for detailed metrics
    metrics_dir = "temp_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Save missing values to CSV - Fixed to ensure arrays have same length
    missing_values_before = missing_values
    missing_values_after = df.isnull().sum()

    # Ensure we're using the same index for both
    all_columns = sorted(
        set(missing_values_before.index) | set(missing_values_after.index)
    )

    missing_values_df = pd.DataFrame(
        {
            "column": all_columns,
            "initial_missing_values": [
                missing_values_before.get(col, 0) for col in all_columns
            ],
            "final_missing_values": [
                missing_values_after.get(col, 0) for col in all_columns
            ],
        }
    )

    missing_values_path = os.path.join(metrics_dir, "missing_values.csv")
    missing_values_df.to_csv(missing_values_path, index=False)
    mlflow.log_artifact(missing_values_path, "missing_values")

    # Log duplicate removal metrics
    mlflow.log_metrics(
        {
            "initial_row_count": initial_row_count,
            "final_row_count": final_row_count,
            "duplicates_removed": initial_row_count - final_row_count,
            "duplicate_removal_percentage": (
                (initial_row_count - final_row_count) / initial_row_count
            )
            * 100,
        }
    )

    # Save column statistics to CSV
    stats_df = df[feature_columns].describe()
    stats_path = os.path.join(metrics_dir, "column_statistics.csv")
    stats_df.to_csv(stats_path)
    mlflow.log_artifact(stats_path, "statistics")

    # Log feature engineering params
    mlflow.log_params(
        {
            "numeric_columns": list(numeric_columns),
            "non_numeric_columns": list(non_numeric_columns),
            "scaled_columns": feature_columns,
            "feature_engineering_steps": [
                "days_since_start",
                "cyclical_month",
                "cyclical_day_of_year",
                "month",
            ],
        }
    )

    # Log final dataset info
    mlflow.log_params(
        {
            "final_rows": len(df),
            "final_columns": len(df.columns),
            "output_file": processed_csv_file,
        }
    )

    # Clean up temporary directory
    import shutil

    shutil.rmtree(metrics_dir)
