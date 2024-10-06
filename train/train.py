import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

# Load the processed data
csv_file = "data/all_cities_processed.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['date'])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

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

# Features and targets
features = [
    'temperature_2m',
    'relative_humidity_2m',
    'dew_point_2m',
    'apparent_temperature',
    'precipitation',
    'wind_speed_10m',
    'pressure_msl',
    'cloud_cover',
]
features = [
    "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm"
]
targets = ['temperature_2m', 'precipitation_probability']

from sklearn.preprocessing import MinMaxScaler
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
city_data[features] = feature_scaler.fit_transform(city_data[features])
city_data[targets] = target_scaler.fit_transform(city_data[targets])

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

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

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
plt.show()

# Plot Precipitation Probability Predictions
plt.figure(figsize=(15, 6))
plt.plot(y_test_inv[:, 1], label='Actual Precipitation Probability')
plt.plot(y_pred_inv[:, 1], label='Predicted Precipitation Probability')
plt.title('Precipitation Probability Prediction vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Precipitation Probability')
plt.legend()
plt.show()

# Save the model (optional)
if not os.path.exists('data/models'):
    os.makedirs('data/models')
model.save('data/models/weather_forecast_lstm.h5')
