import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, abort
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model
model_path = 'data/models/weather_forecast_lstm.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

# Load the scaler
scaler_path = 'data/scalers/minmax_scaler.joblib'
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
feature_scaler = joblib.load(scaler_path)

target_scaler_path = 'data/scalers/target_scaler.joblib'
if not os.path.exists(target_scaler_path):
    raise FileNotFoundError(f"Target scaler file not found: {target_scaler_path}")
target_scaler = joblib.load(target_scaler_path)

# Load the processed data
csv_file = "data/all_cities_processed.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} does not exist.")
df = pd.read_csv(csv_file, parse_dates=['date'])

# Update the features list to match the processed data
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

# Add date-related features separately
date_features = ['days_since_start', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'month']

TIME_STEPS = 24

def create_sequences(data, time_steps=TIME_STEPS):
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

def add_date_features(data):
    data['days_since_start'] = (data['date'] - data['date'].min()).dt.days
    data['month_sin'] = np.sin(2 * np.pi * data['date'].dt.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['date'].dt.month / 12)
    data['day_of_year_sin'] = np.sin(2 * np.pi * data['date'].dt.dayofyear / 365.25)
    data['day_of_year_cos'] = np.cos(2 * np.pi * data['date'].dt.dayofyear / 365.25)
    data['month'] = data['date'].dt.month
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        start_date = request.json['start_date']
        end_date = request.json['end_date']
        city = request.json['city']
        
        # Convert dates to datetime objects (timezone-naive)
        start_date = pd.to_datetime(start_date).tz_localize(None)
        end_date = pd.to_datetime(end_date).tz_localize(None)
        
        # Ensure df['date'] is timezone-naive
        df['date'] = df['date'].dt.tz_localize(None)
        
        # Filter data for the specified city and date range
        city_data = df[(df['city'] == city) & (df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if len(city_data) < TIME_STEPS:
            return jsonify({'error': 'Not enough data for the specified date range'}), 400
        
        # Add date features
        city_data = add_date_features(city_data)
        
        # Prepare input data
        input_data_features = city_data[features+date_features].copy()
        
        # Prepare sequences
        input_sequences = create_sequences(input_data_features.values, time_steps=TIME_STEPS)
        
        # Make predictions
        predictions = model.predict(input_sequences)
        
        # Inverse transform predictions using target_scaler
        predictions_inv = target_scaler.inverse_transform(predictions)
        
        # Prepare the response
        response = {
            'dates': city_data['date'].iloc[TIME_STEPS-1:].dt.strftime('%Y-%m-%d').tolist(),
            'temperature': predictions_inv[:, 0].tolist(),
            'precipitation_probability': predictions_inv[:, 1].tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/data_info', methods=['GET'])
def data_info():
    try:
        city = request.args.get('city')
        if not city:
            abort(400, description="City parameter is required")

        city_data = df[df['city'] == city]
        
        if city_data.empty:
            abort(404, description=f"No data found for city: {city}")

        last_datapoint = city_data.iloc[-1]
        
        response = {
            'city': city,
            'total_datapoints': len(city_data),
            'date_range': {
                'start': city_data['date'].min().strftime('%Y-%m-%d'),
                'end': city_data['date'].max().strftime('%Y-%m-%d')
            },
            'last_datapoint': {
                'date': last_datapoint['date'].strftime('%Y-%m-%d'),
                'temperature': float(last_datapoint['temperature_2m']),
                'precipitation': float(last_datapoint['precipitation'])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
