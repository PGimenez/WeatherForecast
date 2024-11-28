import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
import joblib
import mlflow
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
# For reproducibility
np.random.seed(42)

# Add MLflow setup after the imports
mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
mlflow.set_experiment("MLflow Quickstart 2")

# Load the processed data
csv_file = "data/all_cities_processed.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['date'])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

if not os.path.exists('data/plots'):
    os.makedirs('data/plots')

# Select data for Barcelona
city_name = 'Barcelona'
city_data = df[df['city'] == city_name].copy()
city_data.sort_values('date', inplace=True)
city_data.reset_index(drop=True, inplace=True)

# Check for missing values
city_data.dropna(inplace=True)

# Create 'precipitation_probability' if not present
if 'precipitation_probability' not in city_data.columns:
    city_data['precipitation_probability'] = np.where(city_data['precipitation'] > 0, 1, 0)

# Load the saved scaler
scalers_dir = "data/scalers/"
scaler_filename = os.path.join(scalers_dir, "minmax_scaler.joblib")
feature_scaler = joblib.load(scaler_filename)

# Remove the feature scaling step since data is already scaled
# city_data[features] = feature_scaler.fit_transform(city_data[features])


# Features and targets
features = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    # Add the new date-related features
    "days_since_start", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos", "month"
]
targets = ['temperature_2m', 'precipitation_probability']

# Only scale the target variables
target_scaler = MinMaxScaler()
city_data[targets] = target_scaler.fit_transform(city_data[targets])
target_scaler_filename = os.path.join(scalers_dir, "target_scaler.joblib")
joblib.dump(target_scaler, target_scaler_filename)
print(f"Target scaler saved to {target_scaler_filename}")

# Create sequences
def create_sequences(data, features, targets, time_steps=24):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[features].iloc[i:i+time_steps].values)
        y.append(data[targets].iloc[i+time_steps].values)
    return np.array(X), np.array(y)

TIME_STEPS = 24
X, y = create_sequences(city_data, features, targets, TIME_STEPS)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=y_train.shape[1]))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Modify the model training section
run_name = f"lstm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Log training parameters
    mlflow.log_params({
        "city": city_name,
        "time_steps": TIME_STEPS,
        "train_size": 0.8,
        "lstm_units": 64,
        "dense_units": 32,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "epochs": 20,
        "features": features,
        "targets": targets
    })
    
    # Train model and generate predictions
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Get predictions and calculate metrics
    y_pred = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    
    # Calculate metrics for both targets
    y_test_precip = y_test_inv[:, 1] >= 0.5
    y_pred_precip = y_pred_inv[:, 1] >= 0.5
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # Log all metrics
    mlflow.log_metrics({
        # Training metrics
        "final_train_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        
        # Temperature metrics
        "temperature_mse": mse_temp,
        "temperature_mae": mae_temp,
        "temperature_r2": r2_temp,
        
        # Precipitation probability metrics
        "precip_accuracy": accuracy_score(y_test_precip, y_pred_precip),
        "precip_precision": precision_score(y_test_precip, y_pred_precip),
        "precip_recall": recall_score(y_test_precip, y_pred_precip),
        "precip_f1": f1_score(y_test_precip, y_pred_precip)
    })
    
    # Save plots and log as artifacts
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('data/plots/train_loss.png')
    plt.close()
    
    # Log plots as artifacts
    mlflow.log_artifact('data/plots/train_loss.png', "plots")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

    # Temperature
    mse_temp = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    mae_temp = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    r2_temp = r2_score(y_test_inv[:, 0], y_pred_inv[:, 0])

    print("Temperature Prediction Performance:")
    print(f"MSE: {mse_temp:.2f}")
    print(f"MAE: {mae_temp:.2f}")
    print(f"R^2 Score: {r2_temp:.2f}")

    # Precipitation Probability
    y_test_precip = y_test_inv[:, 1] >= 0.5
    y_pred_precip = y_pred_inv[:, 1] >= 0.5

    print("\nPrecipitation Probability Prediction Performance:")
    print(classification_report(y_test_precip, y_pred_precip, target_names=['No Precipitation', 'Precipitation']))

    # Plot Temperature Predictions
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv[:, 0], label='Actual Temperature')
    plt.plot(y_pred_inv[:, 0], label='Predicted Temperature')
    plt.title('Temperature Prediction vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig('data/plots/temperature_prediction.png')
    plt.close()

    # Plot Precipitation Probability Predictions
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv[:, 1], label='Actual Precipitation Probability')
    plt.plot(y_pred_inv[:, 1], label='Predicted Precipitation Probability')
    plt.title('Precipitation Probability Prediction vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Precipitation Probability')
    plt.legend()
    plt.savefig('data/plots/precipitation_probability_predictions.png')
    plt.close()

    # Log additional prediction plots
    mlflow.log_artifact('data/plots/temperature_prediction.png', "plots")
    mlflow.log_artifact('data/plots/precipitation_probability_predictions.png', "plots")
    
    # Log the model
    mlflow.keras.log_model(model, "model")

# Save the model (optional)
if not os.path.exists('data/models'):
    os.makedirs('data/models')
model.save('data/models/weather_forecast_lstm.h5')
