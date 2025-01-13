import optuna
import mlflow
from datetime import datetime
from train import train, save_and_log_plots
import os

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    params = {
        "lstm_units": trial.suggest_int("lstm_units", 64, 512),
        "dense_units": trial.suggest_int("dense_units", 32, 256),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.3),
        "batch_size": trial.suggest_int("batch_size", 32, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": 10,
        "time_steps": trial.suggest_int("time_steps", 24, 168),
        "train_size": 0.8,
    }

    # Train model with suggested parameters
    model, history, metrics = train(
        data_path="data/all_cities_processed.csv", scalers_dir="data/scalers/", **params
    )

    # Return the validation loss as the objective value to minimize
    return metrics["final_val_loss"]

def perform_hyperparameter_tuning(storage, n_trials):
    """Perform hyperparameter tuning using Optuna."""
    mlflow_callback = MLflowCallback()

    study = optuna.load_study(study_name="weather_forecast", storage=storage)
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])

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
