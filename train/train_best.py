import optuna
import mlflow
from datetime import datetime
from train import train, save_and_log_plots
import os


def train_with_best_params():
    print("Loading best parameters from Optuna study...")

    # Load the study
    storage = "sqlite:////" + "data/optuna/optuna.db"

    # Get the most recent study
    study_summaries = optuna.study.get_all_study_summaries(storage)
    if not study_summaries:
        raise ValueError("No studies found in the database")

    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)

    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    print("\nTraining model with best parameters...")
    mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
    mlflow.set_experiment("weather")

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

        # Save model locally
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        model.save("data/models/weather_forecast_lstm.h5")

        print("\nTraining completed!")
        print("\nFinal metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    train_with_best_params()
