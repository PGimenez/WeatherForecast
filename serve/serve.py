import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import shutil
from fastapi.responses import FileResponse
import tempfile
from pathlib import Path

app = FastAPI()

# Update the CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://weather.carryall.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from MLflow
model_name = "weather_forecast"
model_version = 1
try:
    print(f"\nLoading model '{model_name}' version {model_version} from MLflow...")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"\nFailed to load model from MLflow: {str(e)}\n")
    raise Exception(f"Failed to load model from MLflow: {str(e)}")

# Load the scaler
scaler_path = "data/scalers/feature_scaler.joblib"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
feature_scaler = joblib.load(scaler_path)

target_scaler_path = "data/scalers/target_scaler.joblib"
if not os.path.exists(target_scaler_path):
    raise FileNotFoundError(f"Target scaler file not found: {target_scaler_path}")
target_scaler = joblib.load(target_scaler_path)

# Load the processed data
csv_file = "data/all_cities_processed.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} does not exist.")
df = pd.read_csv(csv_file, parse_dates=["date"])

# Features list should NOT include the target variables
features = [
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

targets = ["temperature_2m", "precipitation"]

TIME_STEPS = 72


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    start_date: str
    end_date: str
    city: str


class PredictionResponse(BaseModel):
    dates: List[str]
    temperature: List[float]
    precipitation: List[float]
    actual_temperature: List[float]
    actual_precipitation: List[float]


class DataInfoResponse(BaseModel):
    city: str
    total_datapoints: int
    date_range: dict
    last_datapoint: dict


# Helper functions remain the same-
def create_sequences(data, time_steps=TIME_STEPS):
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i : (i + time_steps)])
    return np.array(X)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        print(f"\nReceived prediction request:")
        print(f"City: {request.city}")
        print(f"Date range: {request.start_date} to {request.end_date}\n")

        # Convert dates to datetime objects (timezone-naive)
        start_date = pd.to_datetime(request.start_date).tz_localize(None)
        end_date = pd.to_datetime(request.end_date).tz_localize(None)

        # Ensure df['date'] is timezone-naive
        df["date"] = df["date"].dt.tz_localize(None)

        # Filter data for the specified city and date range
        city_data = df[
            (df["city"] == request.city)
            & (df["date"] >= start_date)
            & (df["date"] <= end_date)
        ]

        if len(city_data) < TIME_STEPS:
            raise HTTPException(
                status_code=400, detail="Not enough data for the specified date range"
            )

        # Use all features including target variables for input
        input_data = city_data[features].copy()

        # Create sequences
        input_sequences = create_sequences(input_data, time_steps=TIME_STEPS)

        # Make predictions
        predictions = model.predict(input_sequences)

        # Inverse transform predictions using target_scaler
        predictions_reshaped = predictions.reshape(-1, len(targets))
        predictions_original = target_scaler.inverse_transform(predictions_reshaped)

        # Get actual values for temperature and precipitation
        actual_values = city_data[targets].iloc[TIME_STEPS - 1 :].values

        return {
            "dates": city_data["date"]
            .iloc[TIME_STEPS - 1 :]
            .dt.strftime("%Y-%m-%d %H:00:00")
            .tolist(),
            "temperature": predictions_original[:, 0].tolist(),
            "precipitation": predictions_original[:, 1].tolist(),
            "actual_temperature": actual_values[:, 0].tolist(),
            "actual_precipitation": actual_values[:, 1].tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/data_info", response_model=DataInfoResponse)
async def data_info(city: str):
    try:
        print(f"\nReceived data info request for city: {city}\n")

        if not city:
            raise HTTPException(status_code=400, detail="City parameter is required")

        city_data = df[df["city"] == city]

        if city_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for city: {city}"
            )

        last_datapoint = city_data.iloc[-1]

        return {
            "city": city,
            "total_datapoints": len(city_data),
            "date_range": {
                "start": city_data["date"].min().strftime("%Y-%m-%d"),
                "end": city_data["date"].max().strftime("%Y-%m-%d"),
            },
            "last_datapoint": {
                "date": last_datapoint["date"].strftime("%Y-%m-%d"),
                "temperature": float(last_datapoint["temperature_2m"]),
                "precipitation": float(last_datapoint["precipitation"]),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/download-data")
async def download_data():
    """Download the entire data folder as a zip file"""
    try:
        # Create temporary file with a context manager
        temp_dir = tempfile.mkdtemp()
        try:
            # Create zip file path
            zip_path = Path(temp_dir) / "weather_data.zip"

            # Create zip file from data directory
            data_path = Path("data")
            shutil.make_archive(
                str(zip_path.with_suffix("")),  # Remove .zip as make_archive adds it
                "zip",
                data_path,
            )

            # Return the zip file
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename="weather_data.zip",
                headers={
                    "Content-Disposition": "attachment; filename=weather_data.zip"
                },
                background=None,  # Prevent background task from deleting file too early
            )
        except Exception as e:
            # Clean up temp directory if something goes wrong
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating zip file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",  # Use string reference instead of app instance
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
    )
