import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    classification_report,
)
from mlflow.models.signature import infer_signature
import json
import tempfile
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
from optuna.integration import MLflowCallback
import argparse
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

os.environ["MLFLOW_ARTIFACT_URI"] = "s3://pgimenezbucket/path/"
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

# For reproducibility
np.random.seed(42)

# Add MLflow setup after the imports
# mlflow.create_experiment(
#     "experiment_name2", artifact_location="s3://pgimenezbucket/path/"
# )
mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
# mlflow.create_experiment(
#     "experiment_name4", artifact_location="gs://pgimenezbucket/path/"
# )
mlflow.set_experiment("weather")
# mlflow.set_experiment("MLflow Quickstart 2")

# Load the processed data
csv_file = "data/all_cities_processed.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["date"])
else:
    raise FileNotFoundError(f"{csv_file} does not exist.")

if not os.path.exists("data/plots"):
    os.makedirs("data/plots")

# Select data for Barcelona
city_name = "Barcelona"
city_data = df[df["city"] == city_name].copy()
city_data.sort_values("date", inplace=True)
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


def train(
    data_path: str,
    scalers_dir: str,
    time_steps: int = 24,
    train_size: float = 0.8,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    batch_size: int = 32,
    epochs: int = 5,
):
    """
    Train an LSTM model using Keras TimeseriesGenerator
    """
    # Load only required columns using chunks
    required_cols = ["date", "city"] + features + targets
    chunks = []
    chunk_size = 10000  # Adjust based on your memory constraints

    for chunk in pd.read_csv(
        data_path, usecols=required_cols, parse_dates=["date"], chunksize=chunk_size
    ):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Clean and prepare data
    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)

    # Prepare features and targets
    feature_data = df[features].values
    target_data = df[targets].values

    # Split data
    train_size = int(len(feature_data) * train_size)
    train_features = feature_data[:train_size]
    train_targets = target_data[:train_size]
    test_features = feature_data[train_size:]
    test_targets = target_data[train_size:]

    # Create TimeseriesGenerator for training and testing
    train_generator = TimeseriesGenerator(
        data=train_features,
        targets=train_targets,
        length=time_steps,
        batch_size=batch_size,
    )

    test_generator = TimeseriesGenerator(
        data=test_features,
        targets=test_targets,
        length=time_steps,
        batch_size=batch_size,
    )

    # Load the saved scalers
    target_scaler = joblib.load(os.path.join(scalers_dir, "target_scaler.joblib"))

    # Build model
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(time_steps, len(features))))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=dense_units, activation="relu"))
    model.add(Dense(units=len(targets)))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1,
    )

    # Calculate metrics
    y_pred = model.predict(test_generator)
    # Get the actual test targets, excluding the first time_steps samples
    y_test = test_targets[time_steps:]
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv, history)

    return model, history, metrics


def calculate_metrics(y_test_inv, y_pred_inv, history):
    """Calculate and return all metrics."""
    mse_temp = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    mae_temp = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    r2_temp = r2_score(y_test_inv[:, 0], y_pred_inv[:, 0])

    mse_precip = mean_squared_error(y_test_inv[:, 1], y_pred_inv[:, 1])
    mae_precip = mean_absolute_error(y_test_inv[:, 1], y_pred_inv[:, 1])
    r2_precip = r2_score(y_test_inv[:, 1], y_pred_inv[:, 1])

    return {
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "temperature_mse": mse_temp,
        "temperature_mae": mae_temp,
        "temperature_r2": r2_temp,
        "precipitation_mse": mse_precip,
        "precipitation_mae": mae_precip,
        "precipitation_r2": r2_precip,
    }


def save_and_log_plots(history):
    """Save and log training plots."""
    if not os.path.exists("data/plots"):
        os.makedirs("data/plots")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Test")
    plt.title("Model Loss During Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("data/plots/train_loss.png")
    plt.close()

    mlflow.log_artifact("data/plots/train_loss.png", "plots")


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    params = {
        "lstm_units": trial.suggest_int("lstm_units", 32, 256),
        "dense_units": trial.suggest_int("dense_units", 16, 128),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "batch_size": trial.suggest_int("batch_size", 16, 128),
        "epochs": 20,
        "time_steps": 24,
        "train_size": 0.8,
    }

    # Train model with suggested parameters
    model, history, metrics = train(
        data_path="data/all_cities_processed.csv", scalers_dir="data/scalers/", **params
    )

    # Return the validation loss as the objective value to minimize
    return metrics["final_val_loss"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuning", action="store_true", help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local optimization without distributed storage",
    )
    args = parser.parse_args()

    print("Connecting to MLflow")
    mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
    mlflow.set_experiment("weather")

    if args.tuning:
        print("Performing hyperparameter tuning")
        mlflow_callback = MLflowCallback()

        if args.local:
            # Local optimization without storage
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20, callbacks=[mlflow_callback])

            # Print and train with best parameters for local runs
            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

            # Train final model with best parameters in a new MLflow run
            print("\nTraining final model with best parameters")
            with mlflow.start_run(
                run_name=f"best-params-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ):
                model, history, metrics = train(
                    data_path="data/all_cities_processed.csv",
                    scalers_dir="data/scalers/",
                    **study.best_trial.params,
                )
                # Log best parameters and final metrics
                mlflow.log_params(study.best_trial.params)
                mlflow.log_metrics(metrics)
                save_and_log_plots(history)
                mlflow.keras.log_model(model, "weather_forecast_model")
            print("Saving model")
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            model.save("data/models/weather_forecast_lstm.h5")
        else:
            # Distributed optimization with storage
            storage = os.getenv("OPTUNA_STORAGE")
            if not storage:
                raise ValueError("OPTUNA_STORAGE environment variable not set")

            study = optuna.load_study(study_name="weather_forecast", storage=storage)
            n_trials = int(os.getenv("N_TRIALS", "1"))
            study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

    else:
        print("Training with default parameters")
        with mlflow.start_run(
            run_name=f"lstm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            model, history, metrics = train(
                data_path="data/all_cities_processed.csv",
                scalers_dir="data/scalers/",
            )

            print("Saving model")
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            model.save("data/models/weather_forecast_lstm.h5")

            # Log parameters and metrics
            mlflow.log_params(
                {
                    "time_steps": 24,
                    "train_size": 0.8,
                    "lstm_units": 64,
                    "dense_units": 32,
                    "dropout_rate": 0.2,
                    "batch_size": 32,
                    "epochs": 20,
                    "features": features,
                    "targets": targets,
                }
            )
            mlflow.log_metrics(metrics)
            save_and_log_plots(history)

            # mlflow.keras.log_model(
            #     model,
            #     "weather_forecast_model",
            #     registered_model_name="weather_forecast",
            # )
