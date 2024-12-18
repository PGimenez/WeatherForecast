import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from datetime import datetime

# Load the dataset
csv_file = "data/all_cities.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['date'])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

# Add MLflow setup
mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
mlflow.set_experiment("MLflow Quickstart 2")

# Data Cleaning

# 1. Handle missing values
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
print(f"Numeric columns: {numeric_columns}")
print(f"Non-numeric columns: {non_numeric_columns}")

# Fill missing values for numeric columns with mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values for non-numeric columns with mode (most frequent value)
for col in non_numeric_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2. Correct data types
# Ensure 'date' is datetime and other numerical columns are in correct format
df['date'] = pd.to_datetime(df['date'])
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Remove duplicates
initial_row_count = df.shape[0]
df.drop_duplicates(subset=['date', 'city'], inplace=True)
final_row_count = df.shape[0]
print(f"Removed {initial_row_count - final_row_count} duplicate rows.")

# Feature Engineering

# Convert date to numerical format
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Create cyclical features for month and day of year
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

# Example: Encode 'weather_code' as categorical variable
# if 'weather_code' in df.columns:
#     df['weather_code'] = df['weather_code'].astype('category')
    # Optionally, create dummy variables
    # df = pd.get_dummies(df, columns=['weather_code'])

# Example: Create a 'month' column from 'date'
df['month'] = df['date'].dt.month

# Exploratory Data Analysis (EDA)

# Statistical summaries
print(df.describe())

# # Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Example: Plot temperature over time for a city
# city_name = 'Barcelona'
# city_data = df[df['city'] == city_name]
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=city_data, x='date', y='temperature_2m')
# plt.title(f"Temperature Over Time in {city_name}")
# plt.xlabel("Date")
# plt.ylabel("Temperature (°C)")
# plt.show()

# Data Transformation

# Example: Normalize numerical features
from sklearn.preprocessing import MinMaxScaler

# Create the scalers directory if it doesn't exist
scalers_dir = "data/scalers/"
os.makedirs(scalers_dir, exist_ok=True)

scaler = MinMaxScaler()
numeric_columns_to_scale = [col for col in numeric_columns if col not in ['days_since_start', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']]
df[numeric_columns_to_scale] = scaler.fit_transform(df[numeric_columns_to_scale])

# Save the scaler
scaler_filename = os.path.join(scalers_dir, "minmax_scaler.joblib")
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Save the processed data
processed_csv_file = "data/all_cities_processed.csv"
df.to_csv(processed_csv_file, index=False)
print(f"Processed data saved to {processed_csv_file}.")

run_name = f"data-processing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Log basic parameters (keep these as regular params since they're important)
    mlflow.log_params({
        "input_file": csv_file,
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "data_start_date": df['date'].min().strftime('%Y-%m-%d'),
        "data_end_date": df['date'].max().strftime('%Y-%m-%d'),
        "num_cities": len(df['city'].unique())
    })
    
    # Create temporary CSV files for detailed metrics
    metrics_dir = "temp_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save missing values to CSV
    missing_values_df = pd.DataFrame({
        'column': missing_values.index,
        'initial_missing_values': missing_values.values,
        'final_missing_values': df.isnull().sum().values
    })
    missing_values_path = os.path.join(metrics_dir, "missing_values.csv")
    missing_values_df.to_csv(missing_values_path, index=False)
    mlflow.log_artifact(missing_values_path, "missing_values")
    
    # Log duplicate removal metrics (keep these as regular metrics since they're important summary stats)
    mlflow.log_metrics({
        "initial_row_count": initial_row_count,
        "final_row_count": final_row_count,
        "duplicates_removed": initial_row_count - final_row_count,
        "duplicate_removal_percentage": ((initial_row_count - final_row_count) / initial_row_count) * 100
    })
    
    # Save column statistics to CSV
    stats_df = df[numeric_columns_to_scale].describe()
    stats_path = os.path.join(metrics_dir, "column_statistics.csv")
    stats_df.to_csv(stats_path)
    mlflow.log_artifact(stats_path, "statistics")
    
    # Log feature engineering params
    mlflow.log_params({
        "numeric_columns": list(numeric_columns),
        "non_numeric_columns": list(non_numeric_columns),
        "scaled_columns": numeric_columns_to_scale,
        "feature_engineering_steps": [
            "days_since_start",
            "cyclical_month",
            "cyclical_day_of_year",
            "month"
        ]
    })
    
    # Log final dataset info
    mlflow.log_params({
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "output_file": processed_csv_file
    })
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(metrics_dir)
