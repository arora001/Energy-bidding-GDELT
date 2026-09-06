
# AI Energy Load Forecasting

**AI-Energy-Load-New** is a Python-based pipeline for forecasting energy load using a combination of news (via GDELT GKG), weather data, and historical load patterns. It supports data ingestion, preprocessing, feature engineering, model training (XGBoost & LSTM), hyperparameter tuning with Optuna, walk-forward evaluation, outlier analysis, and visualization.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Flow](#data-flow)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Project Overview

 This project implements a comprehensive workflow to forecast energy load (power demand) for the Delhi region by combining:

1. **News Data**: The GDELT Global Knowledge Graph (GKG) pipeline extracts thematic features (intensity, evolution, occurrence, co-occurrence).
2. **Weather Data**: Aggregated weather features (temperature, humidity, wind, precipitation).
3. **Historical Load**: Past power demand lags and engineered temporal features.

 Two model families are supported:

- XGBoost regression with Optuna hyperparameter tuning
- LSTM neural networks with walk-forward hyperparameter optimization

 Evaluation uses walk-forward cross-validation, outlier detection & analysis, and extensive plotting.

## Repository Structure

```
 .
 ├── config.py                         # Central configuration and directory setup
 ├── Correlation_and_preprocessing/    # Data merging and correlation analysis
 ├── feature_engineering/              # Modular feature engineering scripts
 ├── GDELT_gkg/                        # Scripts to collect and process GDELT GKG news data
 ├── Hard_code_plots/                  # Ad-hoc plotting scripts
 ├── Model_pedictor/                   # Data preparation, XGBoost modeling, and evaluation utilities
 ├── MODELS/                           # Pre-trained or serialized model files
 ├── MODEL_RESULTS/                    # Saved LSTM walk-forward model splits and metrics
 ├── OUTPUT_DIR/                       # Generated data, models, logs, figures, and intermediate outputs
 ├── Weather_Energy/                   # Weather data aggregation and energy-weather merge scripts
 ├── *.py (top-level)                  # Orchestration and pipeline entry-point scripts
 ├── README.md                         # This documentation
 └── .gitignore                        # Git ignore patterns
```

## Data Flow

1. **Data Ingestion**:
   - GDELT GKG: `GDELT_gkg/gkg_data_gathering.py`, `gkg_data_aggregation.py`, `gkg_data_sparsing.py`, `gkg_data_theme_analysis.py`
   - Weather/Energy: `Weather_Energy/weather_aggregation.py`
2. **Data Preprocessing & Merging**:
   - `Correlation_and_preprocessing/merge.py`
   - Missing value handling and outlier removal: `Model_pedictor/preparation.py`
3. **Correlation Analysis**:
   - `Correlation_and_preprocessing/Correalation.py`
4. **Feature Engineering**:
   - `feature_engineering/` modules: `theme_intensity`, `theme_evolution`, `theme_occurence`, `temporal_effects`
5. **Data Splitting**:
   - `prepare_splits.py` for train/test split generation (walk-forward)
6. **Model Training & Tuning**:
   - XGBoost + Optuna: `run_xgb_optuna.py`
   - LSTM + Optuna: `lstm_walkforward_eval.py`
7. **Walk-Forward Evaluation**:
   - `orchestrate_walk_forward.py` & `orchestrate_walk_forward_just_weather.py`
   - Evaluation metrics and summaries saved under `OUTPUT_DIR/running_eval` and `MODEL_RESULTS/`
8. **Outlier Detection**:
   - `outlier_detection_and_evaluation.py` -> results in `outlier_results/`
9. **Visualization**:
   - Analysis & plotting scripts under `Hard_code_plots/` and generated figures in `OUTPUT_DIR/figures`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/AI-Energy-Load-New.git
   cd AI-Energy-Load-New
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- Review and adjust settings in `config.py` (e.g., data paths, date ranges, forecasting horizon).
- Ensure directory structure is initialized:
  ```bash
  python - << 'EOF'
  from config import setup_and_verify
  setup_and_verify()
  EOF
  ```

## Usage

- **Ingest & preprocess news data**:
  ```bash
  python GDELT_gkg/gkg_data_gathering.py
  python GDELT_gkg/gkg_data_aggregation.py
  python GDELT_gkg/gkg_data_sparsing.py
  python GDELT_gkg/gkg_data_theme_analysis.py
  ```
- **Prepare weather data**:
  ```bash
  python Weather_Energy/weather_aggregation.py
  ```
- **Merge datasets & analyze correlations**:
  ```bash
  python Correlation_and_preprocessing/merge.py
  python Correlation_and_preprocessing/Correalation.py
  ```
- **Generate features**:
  ```bash
  python feature_engineering/...
  ```
- **Split data for modeling**:
  ```bash
  python prepare_splits.py
  ```
- **Run hyperparameter tuning & training**:
  ```bash
  python run_xgb_optuna.py
  python lstm_walkforward_eval.py
  ```
- **Perform walk-forward evaluation**:
  ```bash
  python orchestrate_walk_forward.py
  ```
- **Detect outliers**:
  ```bash
  python outlier_detection_and_evaluation.py
  ```

## Dependencies

- Python 3.8+
- pandas, numpy, scipy, scikit-learn, xgboost, tensorflow, keras, optuna, matplotlib, seaborn, joblib, statsmodels

## Contributing

 Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.
