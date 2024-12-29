import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import mlflow
from datetime import datetime

# Import the features list from train.py
from train import features

def load_and_prepare_data(data_path, scalers_dir, time_steps=24, test_size=0.2):
    """Load and prepare data for testing"""
    # Load data
    df = pd.read_csv(data_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)

    # Get feature data
    feature_data = df[features].values
    
    # Split into train/test
    train_size = int(len(feature_data) * (1 - test_size))
    test_features = feature_data[train_size:-1]  # Remove last entry
    test_targets = feature_data[train_size+1:]   # Shift by 1 to predict next timestep
    test_dates = df["date"][train_size+1:].reset_index(drop=True)

    # Create TimeseriesGenerator for testing
    test_generator = TimeseriesGenerator(
        data=test_features,
        targets=test_targets,
        length=time_steps,
        batch_size=1  # Use batch_size=1 for detailed testing
    )

    return test_generator, test_targets[time_steps:], test_dates[time_steps:]

def evaluate_predictions(y_true, y_pred, dates, feature_names):
    """Evaluate and visualize predictions"""
    results = {}
    
    # Create directory for plots if it doesn't exist
    plots_dir = "data/test_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create temporary directory for MLflow artifacts
    temp_metrics_dir = "temp_metrics"
    os.makedirs(temp_metrics_dir, exist_ok=True)
    
    # Calculate metrics for each feature
    feature_metrics = []
    for i, feature in enumerate(feature_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        results[feature] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        # Create plot for each feature
        plt.figure(figsize=(15, 6))
        plt.plot(dates, y_true[:, i], label='Actual', alpha=0.7)
        plt.plot(dates, y_pred[:, i], label='Predicted', alpha=0.7)
        plt.title(f'{feature} - Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot and collect metrics
        plot_path = f'{plots_dir}/{feature}_predictions.png'
        plt.savefig(plot_path)
        plt.close()
        
        feature_metrics.append({
            'feature': feature,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
    
    # Save metrics to CSV for MLflow
    metrics_df = pd.DataFrame(feature_metrics)
    metrics_path = os.path.join(temp_metrics_dir, "feature_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Create heatmap of feature-wise R2 scores
    plt.figure(figsize=(12, 8))
    r2_scores = [results[feature]['R2'] for feature in feature_names]
    sns.barplot(x=r2_scores, y=feature_names)
    plt.title('R² Score by Feature')
    plt.xlabel('R² Score')
    plt.tight_layout()
    plt.savefig('data/test_plots/r2_scores.png')
    plt.close()

    return results, plots_dir, temp_metrics_dir

def main():
    # MLflow setup
    mlflow.set_tracking_uri(uri="http://mlflow.carryall.local:80")
    mlflow.set_experiment("weather")
    
    run_name = f"model-testing-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Load model and scaler
        model_path = "data/models/weather_forecast_lstm.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = load_model(model_path)
        feature_scaler = joblib.load("data/scalers/feature_scaler.joblib")

        # Load and prepare test data
        test_generator, y_test, test_dates = load_and_prepare_data(
            data_path="data/all_cities_processed.csv",
            scalers_dir="data/scalers/",
            time_steps=24
        )

        # Log model parameters
        mlflow.log_params({
            'time_steps': 24,
            'test_size': 0.2,
            'model_path': model_path,
            'feature_count': len(features)
        })
        
        # Make predictions and evaluate
        print("Making predictions...")
        y_pred = model.predict(test_generator)
        y_test_inv = feature_scaler.inverse_transform(y_test)
        y_pred_inv = feature_scaler.inverse_transform(y_pred)
        
        print("Evaluating predictions...")
        results, plots_dir, temp_metrics_dir = evaluate_predictions(
            y_test_inv, y_pred_inv, test_dates, features
        )
        
        # Calculate and log average metrics
        avg_mse = np.mean([results[f]['MSE'] for f in features])
        avg_mae = np.mean([results[f]['MAE'] for f in features])
        avg_r2 = np.mean([results[f]['R2'] for f in features])
        
        mlflow.log_metrics({
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            'avg_r2': avg_r2
        })
        
        # Log individual feature metrics
        for feature in features:
            mlflow.log_metrics({
                f'{feature}_mse': results[feature]['MSE'],
                f'{feature}_mae': results[feature]['MAE'],
                f'{feature}_r2': results[feature]['R2']
            })
        
        # Log artifacts
        mlflow.log_artifacts(plots_dir, "prediction_plots")
        mlflow.log_artifacts(temp_metrics_dir, "metrics")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_metrics_dir)
        
        # Print summary statistics
        print("\nModel Performance Summary:")
        print("-" * 50)
        
        # Calculate average metrics across all features
        avg_mse = np.mean([results[f]['MSE'] for f in features])
        avg_mae = np.mean([results[f]['MAE'] for f in features])
        avg_r2 = np.mean([results[f]['R2'] for f in features])
        
        print(f"Average MSE across all features: {avg_mse:.4f}")
        print(f"Average MAE across all features: {avg_mae:.4f}")
        print(f"Average R² across all features: {avg_r2:.4f}")
        
        print("\nTop 5 Best Predicted Features:")
        r2_scores = {f: results[f]['R2'] for f in features}
        top_features = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, r2 in top_features:
            print(f"{feature:30} R² = {r2:.4f}")
        
        print("\nBottom 5 Predicted Features:")
        bottom_features = sorted(r2_scores.items(), key=lambda x: x[1])[:5]
        for feature, r2 in bottom_features:
            print(f"{feature:30} R² = {r2:.4f}")

if __name__ == "__main__":
    main() 