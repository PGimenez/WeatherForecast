import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def build_model(time_steps, features, lstm_units=128, dense_units=64, dropout_rate=0.1, learning_rate=0.01):
    """Build and compile the LSTM model"""
    model = Sequential([
        LSTM(units=lstm_units, input_shape=(time_steps, len(features)), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=dense_units, activation='relu'),
        Dense(units=len(features))  # Output layer predicts all features
    ])

    # Create learning rate schedule
    initial_learning_rate = learning_rate
    decay_steps = 1000  # Adjust based on your dataset size
    decay_rate = 0.9  # decay by 10% each time

    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Use the schedule in the optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error'
    )

    return model
