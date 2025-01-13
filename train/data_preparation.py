import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import os

def load_and_prepare_data(data_path, scalers_dir, time_steps=24, test_size=0.2):
    """Load and prepare data for training"""
    # Load data
    df = pd.read_csv(data_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)

    # Get feature data
    feature_data = df[features].values
    
    # Split into train/test
    train_size = int(len(feature_data) * (1 - test_size))
    train_features = feature_data[:train_size]
    train_targets = feature_data[1:train_size + 1]  # Shift by 1 to predict next timestep
    test_features = feature_data[train_size:-1]  # Remove last entry
    test_targets = feature_data[train_size + 1:]  # Shift by 1 to predict next timestep

    return train_features, train_targets, test_features, test_targets

def create_timeseries_generator(features, targets, time_steps=24, batch_size=32):
    """Create TimeseriesGenerator for training and testing"""
    generator = TimeseriesGenerator(
        data=features,
        targets=targets,
        length=time_steps,
        batch_size=batch_size,
    )
    return generator
