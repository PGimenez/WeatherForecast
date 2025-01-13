import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import joblib
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import json
import tempfile
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def train_model(data_path, scalers_dir, time_steps=24, train_size=0.8, lstm_units=128, dense_units=64, dropout_rate=0.1, batch_size=32, epochs=10, learning_rate=0.01):
    """Train an LSTM model using Keras TimeseriesGenerator to predict all features"""
    # Load only required columns using chunks
    required_cols = ["date", "city"] + features + targets
    chunks = []
    chunk_size = 10000  # Adjust based on your memory constraints

    for chunk in pd.read_csv(data_path, usecols=required_cols, parse_dates=["date"], chunksize=chunk_size):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Clean and prepare data
    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)

    # Load the feature scaler instead of target scaler
    feature_scaler = joblib.load(os.path.join(scalers_dir, "feature_scaler.joblib"))

    # Prepare features and targets (now they're the same)
    feature_data = df[features].values
    target_data = feature_data  # We want to predict all features

    # Split data
    train_size = int(len(feature_data) * train_size)
    train_features = feature_data[:train_size]
    train_targets = target_data[1:train_size + 1]  # Shift by 1 to predict next timestep
    test_features = feature_data[train_size:-1]  # Remove last entry
    test_targets = target_data[train_size + 1:]  # Shift by 1 to predict next timestep

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

    # Build and train model
    model = Sequential(
        [
            LSTM(
                units=lstm_units,
                input_shape=(time_steps, len(features)),
                return_sequences=True,
            ),
            Dropout(dropout_rate),
            LSTM(units=lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=dense_units, activation="relu"),
            Dense(units=len(features)),  # Output layer predicts all features
        ]
    )

    # Create learning rate schedule
    initial_learning_rate = learning_rate
    decay_steps = len(train_generator)  # decay once per epoch
    decay_rate = 0.9  # decay by 10% each time

    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )

    # Use the schedule in the optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="mean_squared_error",
    )

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

    # Use feature_scaler for inverse transform since we're predicting all features
    y_test_inv = feature_scaler.inverse_transform(y_test)
    y_pred_inv = feature_scaler.inverse_transform(y_pred)

    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv, history)

    return model, history, metrics

def calculate_metrics(y_test_inv, y_pred_inv, history):
    """Calculate and return metrics for all features."""
    metrics = {
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
    }

    # Calculate MSE, MAE, and R2 for each feature
    for i, feature in enumerate(features):
        mse = mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])
        mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
        r2 = r2_score(y_test_inv[:, i], y_pred_inv[:, i])

        metrics[f"{feature}_mse"] = mse
        metrics[f"{feature}_mae"] = mae
        metrics[f"{feature}_r2"] = r2

    return metrics

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
