# Weather forecasting ML pipeline

## Overview
A comprehensive machine learning pipeline for multi-feature weather forecasting using LSTM and XGBoost models. The project implements a full MLOps lifecycle, from data processing to model deployment, with real-time predictions served through a REST API, and a web UI.

## Features
- Multi-model approach (LSTM and XGBoost) for weather prediction
- Extensive EDA and data visualization
- MLflow experiment tracking and model registry
- Argo Workflows for model training and data gathering
- FastAPI-based model serving
- Vuejs web UI
- Automated hyperparameter optimization with Optuna
- GPU support for accelerated training

## Key Components

#### Data Processing
- Handles missing values and outliers
- Feature engineering and scaling
- Time series data preparation

#### Exploratory Data Analysis
- Comprehensive statistical analysis
- Time series visualizations
- Feature correlation analysis
- Weather pattern visualization

#### Model Training
- LSTM model with hyperparameter optimization
- XGBoost model for comparison
- MLflow experiment tracking

#### Model Serving
- FastAPI REST API
- Real-time predictions
- Multi-model inference

#### Frontend
- Vue.js based web interface
- Real-time weather predictions visualization
- Interactive charts and graphs

