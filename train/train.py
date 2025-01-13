import os
import argparse
import mlflow
from datetime import datetime
from optuna.integration import MLflowCallback
from data_preparation import load_and_prepare_data, create_timeseries_generator
from model import build_model
from training import train_model, save_and_log_plots
from optuna_tuning import perform_hyperparameter_tuning

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
                model, history, metrics = train_model(
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

            perform_hyperparameter_tuning(storage, n_trials=int(os.getenv("N_TRIALS", "1")))

    else:
        print("Training with default parameters")
        with mlflow.start_run(
            run_name=f"lstm-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            model, history, metrics = train_model(
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
                    "lstm_units": 128,
                    "dense_units": 64,
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
