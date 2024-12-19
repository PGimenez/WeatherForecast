import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
import joblib
import mlflow
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from mlflow.models.signature import infer_signature
import json
import tempfile
import os
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

# Load the saved scalers
scalers_dir = "data/scalers/"
target_scaler = joblib.load(os.path.join(scalers_dir, "target_scaler.joblib"))

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
    "soil_moisture_100_to_255cm"
]

targets = ['temperature_2m', 'precipitation']

# Create sequences
def create_sequences(data, input_features, targets, time_steps=24):
    X = []
    y = []
    
    for i in range(len(data) - time_steps):
        # Create sequence from input features
        sequence = data[input_features].iloc[i:i+time_steps].values
        X.append(sequence)
        # Get target values
        target = data[targets].iloc[i+time_steps].values
        y.append(target)
    
    return np.array(X), np.array(y)

TIME_STEPS = 24
# Use all features (weather + date) for input
X, y = create_sequences(city_data, features, targets, TIME_STEPS)

# Network Parameters
TRAIN_SIZE = 0.8
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 20

# Split data
train_size = int(len(X) * TRAIN_SIZE)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(units=DENSE_UNITS, activation='relu'))
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
        "train_size": TRAIN_SIZE,
        "lstm_units": LSTM_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "features": features,
        "targets": targets
    })
    
    # Train model and generate predictions
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Get predictions and calculate metrics
    y_pred = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    
    # Calculate temperature metrics
    mse_temp = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    mae_temp = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    r2_temp = r2_score(y_test_inv[:, 0], y_pred_inv[:, 0])
    
    print("Temperature Prediction Performance:")
    print(f"MSE: {mse_temp:.2f}")
    print(f"MAE: {mae_temp:.2f}")
    print(f"R^2 Score: {r2_temp:.2f}")

    # Precipitation metrics (changed from probability to amount)
    mse_precip = mean_squared_error(y_test_inv[:, 1], y_pred_inv[:, 1])
    mae_precip = mean_absolute_error(y_test_inv[:, 1], y_pred_inv[:, 1])
    r2_precip = r2_score(y_test_inv[:, 1], y_pred_inv[:, 1])

    print("\nPrecipitation Amount Prediction Performance:")
    print(f"MSE: {mse_precip:.2f}")
    print(f"MAE: {mae_precip:.2f}")
    print(f"R^2 Score: {r2_precip:.2f}")

    # Update MLflow metrics
    mlflow.log_metrics({
        # Training metrics
        "final_train_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        
        # Temperature metrics
        "temperature_mse": mse_temp,
        "temperature_mae": mae_temp,
        "temperature_r2": r2_temp,
        
        # Precipitation amount metrics (updated)
        "precipitation_mse": mse_precip,
        "precipitation_mae": mae_precip,
        "precipitation_r2": r2_precip
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
    
    # Simply log the model without input example
    mlflow.keras.log_model(
        model, 
        "weather_forecast_model"
    )

# Save the model (optional)
if not os.path.exists('data/models'):
    os.makedirs('data/models')
model.save('data/models/weather_forecast_lstm.h5')
