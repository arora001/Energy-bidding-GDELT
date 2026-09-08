import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
from pathlib import Path
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'xgboost_model.log'))
    ]
)
logger = logging.getLogger(__name__)

# Create robust path configuration
try:
    # Try to load from config file first
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import PATHS
    logger.info("Loaded paths from config.py")
except (ImportError, AttributeError):
    logger.warning("Could not import paths from config.py, using default paths")
    # Default project paths based on current directory structure
    BASE_DIR = Path(r'C:\Users\nikun\Desktop\MLPR\AI-Energy-Load-New')
    PATHS = {
        "SPLIT_DATA_DIR": str(BASE_DIR / "DATA" / "split_data"),
        "MODELS_DIR": str(BASE_DIR / "MODELS"),
        "MODEL_EVAL_DIR": str(BASE_DIR / "MODEL_EVALUATION"),
        "OUTPUT_DIR": str(BASE_DIR / "OUTPUT_DIR"),
    }

# Define paths 
SPLIT_DATA_DIR = Path(PATHS["SPLIT_DATA_DIR"])
MODELS_DIR = Path(PATHS["MODELS_DIR"])
EVAL_DIR = Path(PATHS["MODEL_EVAL_DIR"])

# Ensure directories exist
os.makedirs(MODELS_DIR / 'xgboost', exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

logger.info(f"Split data directory: {SPLIT_DATA_DIR}")
logger.info(f"Models directory: {MODELS_DIR}")
logger.info(f"Evaluation directory: {EVAL_DIR}")

def load_split_data(strategy="month_based"):
    """Load the most recent split data files for the specified strategy"""
    logger.info(f"Loading split data for '{strategy}' strategy...")
    
    # Find most recent train and test files
    train_dir = SPLIT_DATA_DIR / "train_data"
    test_dir = SPLIT_DATA_DIR / "test_data"
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Alternative locations if not found in expected directories
    alternative_locations = [
        Path(PATHS.get("OUTPUT_DIR", "")) / "split_data",
        Path(r'C:\Users\nikun\Desktop\MLPR\AI-Energy-Load-New\OUTPUT_DIR\split_data')
    ]
    
    for alt_loc in alternative_locations:
        alt_train_dir = alt_loc / "train_data"
        alt_test_dir = alt_loc / "test_data"
        
        if os.path.exists(alt_train_dir) and os.path.exists(alt_test_dir):
            logger.info(f"Using alternative data location: {alt_loc}")
            train_dir = alt_train_dir
            test_dir = alt_test_dir
            break
    
    # Get list of files matching the strategy
    try:
        train_files = [f for f in os.listdir(train_dir) if f.startswith(f"train_data_{strategy}_")]
        test_files = [f for f in os.listdir(test_dir) if f.startswith(f"test_data_{strategy}_")]
    except FileNotFoundError:
        logger.error(f"Directory not found: {train_dir} or {test_dir}")
        # Try a more exhaustive search
        logger.info("Searching for data files in alternate locations...")
        
        for root, dirs, files in os.walk(Path(r'C:\Users\nikun\Desktop\MLPR\AI-Energy-Load-New')):
            for file in files:
                if file.startswith(f"train_data_{strategy}_") or file.startswith(f"test_data_{strategy}_"):
                    logger.info(f"Found data file: {os.path.join(root, file)}")
        
        raise FileNotFoundError(f"No split data files found for strategy '{strategy}'")
    
    if not train_files or not test_files:
        raise FileNotFoundError(f"No split data files found for strategy '{strategy}'")
    
    # Sort by timestamp (newest first)
    train_files.sort(reverse=True)
    test_files.sort(reverse=True)
    
    # Load the most recent files
    train_path = train_dir / train_files[0]
    test_path = test_dir / test_files[0]
    
    logger.info(f"Loading training data from: {train_files[0]}")
    logger.info(f"Loading testing data from: {test_files[0]}")
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Identify datetime column and set as index if not already
        datetime_cols = [col for col in train_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols and not pd.api.types.is_datetime64_dtype(train_df.index):
            date_col = datetime_cols[0]
            logger.info(f"Setting {date_col} as index")
            train_df[date_col] = pd.to_datetime(train_df[date_col])
            test_df[date_col] = pd.to_datetime(test_df[date_col])
            train_df.set_index(date_col, inplace=True)
            test_df.set_index(date_col, inplace=True)
        
        # Identify target column
        target_candidates = ['target', 'power_demand', 'load', 'energy_load']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in train_df.columns:
                target_col = candidate
                logger.info(f"Found target column: {target_col}")
                break
        
        if target_col is None:
            logger.warning("Target column not explicitly found, using last column as target")
            target_col = train_df.columns[-1]
        
        # Separate features and target
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
        
        logger.info(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")
        logger.info(f"Features: {len(X_train.columns)}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def add_time_features(X_train, X_test):
    """Add time-based features to help XGBoost capture temporal patterns"""
    logger.info("Adding time-based features...")
    
    # Make copies to avoid modifying originals
    X_train_time = X_train.copy()
    X_test_time = X_test.copy()
    
    # Check if index is datetime
    if not isinstance(X_train_time.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, attempting to convert...")
        try:
            X_train_time.index = pd.to_datetime(X_train_time.index)
            X_test_time.index = pd.to_datetime(X_test_time.index)
        except:
            logger.error("Could not convert index to datetime. Using numerical features only.")
            return X_train, X_test
    
    # Add hour of day
    X_train_time['hour'] = X_train_time.index.hour
    X_test_time['hour'] = X_test_time.index.hour
    
    # Add day of week (0=Monday, 6=Sunday)
    X_train_time['dayofweek'] = X_train_time.index.dayofweek
    X_test_time['dayofweek'] = X_test_time.index.dayofweek
    
    # Add month
    X_train_time['month'] = X_train_time.index.month
    X_test_time['month'] = X_test_time.index.month
    
    # Add day of year
    X_train_time['dayofyear'] = X_train_time.index.dayofyear
    X_test_time['dayofyear'] = X_test_time.index.dayofyear
    
    # Add is_weekend flag
    X_train_time['is_weekend'] = (X_train_time.index.dayofweek >= 5).astype(int)
    X_test_time['is_weekend'] = (X_test_time.index.dayofweek >= 5).astype(int)
    
    # Add quarter of day (0-3)
    X_train_time['quarter_of_day'] = X_train_time.index.hour // 6
    X_test_time['quarter_of_day'] = X_test_time.index.hour // 6
    
    # Create cyclical features for hour to capture daily patterns
    X_train_time['hour_sin'] = np.sin(2 * np.pi * X_train_time.index.hour / 24)
    X_train_time['hour_cos'] = np.cos(2 * np.pi * X_train_time.index.hour / 24)
    X_test_time['hour_sin'] = np.sin(2 * np.pi * X_test_time.index.hour / 24)
    X_test_time['hour_cos'] = np.cos(2 * np.pi * X_test_time.index.hour / 24)
    
    # Create cyclical features for day of year to capture yearly patterns
    X_train_time['day_of_year_sin'] = np.sin(2 * np.pi * X_train_time.index.dayofyear / 365)
    X_train_time['day_of_year_cos'] = np.cos(2 * np.pi * X_train_time.index.dayofyear / 365)
    X_test_time['day_of_year_sin'] = np.sin(2 * np.pi * X_test_time.index.dayofyear / 365)
    X_test_time['day_of_year_cos'] = np.cos(2 * np.pi * X_test_time.index.dayofyear / 365)
    
    logger.info(f"Added time features. New feature count: {len(X_train_time.columns)}")
    
    return X_train_time, X_test_time

def compute_metrics(y_true, y_pred):
    """Return key regression metrics as a dictionary."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mean_error': (y_true - y_pred).mean(),
        'median_error': (y_true - y_pred).median(),
        'max_error': (y_true - y_pred).max(),
        'min_error': (y_true - y_pred).min(),
        'mean_abs_error': np.abs(y_true - y_pred).mean(),
        'mean_percent_error': ((y_true - y_pred) / y_true * 100).mean(),
        'median_percent_error': ((y_true - y_pred) / y_true * 100).median(),
    }

def train_xgboost_model(X_train, y_train, X_val, y_val, model_name=None, params=None):
    """Train an XGBoost model for energy load prediction"""
    logger.info("Training XGBoost model...")

    if params is None:
        # Default parameters (used if Optuna not involved)
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'n_estimators': 1000,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 50,
            'seed': 42
        }

    eval_set = [(X_train.values, y_train.values), (X_val.values, y_val.values)]

    model = xgb.XGBRegressor(**params)

    logger.info("Starting model training...")
    model.fit(
        X_train.values, y_train.values,
        eval_set=eval_set,
        verbose=100
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best validation RMSE: {model.best_score:.4f}")

    if model_name:
        model_path = MODELS_DIR / 'xgboost' / f'{model_name}.json'
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save feature importance
        try:
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            importance_path = MODELS_DIR / 'xgboost' / f'{model_name}_feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)

            logger.info("\nTop 10 important features:")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"{row['Feature']}: {row['Importance']:.6f}")
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {e}")
            importance_df = None
    else:
        importance_df = None

    return model, importance_df


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and generate visualizations."""
    logger.info("Evaluating model performance.")

    eval_dir = EVAL_DIR / f'xgboost_{model_name}'
    os.makedirs(eval_dir, exist_ok=True)

    y_pred = model.predict(X_test.values)

    # Build prediction DataFrame
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    }, index=X_test.index)
    predictions_df['error'] = predictions_df['actual'] - predictions_df['predicted']
    predictions_df['abs_error'] = np.abs(predictions_df['error'])
    predictions_df['percent_error'] = (predictions_df['error'] / predictions_df['actual']) * 100

    # Save predictions
    predictions_df.to_csv(eval_dir / 'predictions.csv')

    # Compute and save metrics
    metrics = compute_metrics(y_test, y_pred)
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Log metrics
    logger.info("\nModel Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    # Plot
    create_evaluation_plots(predictions_df, eval_dir, model_name)

    return predictions_df, metrics

def create_evaluation_plots(predictions_df, eval_dir, model_name, save_plots=True):
    """Create and optionally save evaluation plots"""
    plots_dir = eval_dir / 'plots'
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['actual'], 'b-', label='Actual', alpha=0.7)
    plt.plot(predictions_df.index, predictions_df['predicted'], 'r-', label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted - XGBoost {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Power Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / 'actual_vs_predicted.png', dpi=300)
    plt.close()

    # Plot 2: Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Error Distribution - XGBoost {model_name}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / 'error_distribution.png', dpi=300)
    plt.close()

    # Plot 3: Error over Time
    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df.index, predictions_df['error'], 'g-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Error over Time - XGBoost {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / 'error_over_time.png', dpi=300)
    plt.close()

    # Plot 4: Actual vs Predicted Scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.5)
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'Actual vs Predicted Scatter - XGBoost {model_name}')
    plt.xlabel('Actual Power Demand')
    plt.ylabel('Predicted Power Demand')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    if save_plots:
        plt.savefig(plots_dir / 'actual_vs_predicted_scatter.png', dpi=300)
    plt.close()

    # Plot 5: Feature Importance
    importance_path = MODELS_DIR / 'xgboost' / f'{model_name}_feature_importance.csv'
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        top_features = importance_df.head(20)

        plt.figure(figsize=(12, 10))
        plt.barh(range(len(top_features)), top_features['Importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.title('Top 20 Feature Importance - XGBoost')
        plt.xlabel('Importance')
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / 'feature_importance.png', dpi=300)
        plt.close()
    else:
        logger.warning("Feature importance CSV not found — skipping plot.")

    if save_plots:
        logger.info(f"Evaluation plots saved to {plots_dir}")

def main():
    """Main execution function"""
    logger.info("==== Energy Load XGBoost Model Training ====")
    
    # Check for available strategy files first
    available_strategies = []
    train_dir = None
    
    # Look in possible data locations
    possible_locations = [
        SPLIT_DATA_DIR / "train_data",
        Path(PATHS.get("OUTPUT_DIR", "")) / "split_data" / "train_data"
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            train_dir = loc
            # Get strategy names from existing files
            for file in os.listdir(train_dir):
                if file.startswith("train_data_"):
                    # Extract strategy name from filename pattern train_data_STRATEGY_*
                    parts = file.split('_')
                    if len(parts) >= 3:
                        strategy = parts[2]
                        if strategy not in available_strategies:
                            available_strategies.append(strategy)
            if available_strategies:
                break
    
    if available_strategies:
        logger.info(f"Found data for these strategies: {available_strategies}")
        strategies = available_strategies
    else:
        # No files found, use the preparation.py naming convention
        strategies = ['month_based', 'fully_random', 'seasonal_block']
        logger.warning(f"No strategy files found. Will try: {strategies}")
    
    # Check if command line arguments were provided for strategies
    if len(sys.argv) > 1:
        strategies = sys.argv[1:]
        logger.info(f"Using command line specified strategies: {strategies}")
    else:
        logger.info(f"Using default strategies: {strategies}")
    
    for strategy in strategies:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {strategy.upper()} split strategy")
            logger.info(f"{'='*50}")
            
            # Load split data
            X_train, y_train, X_test, y_test = load_split_data(strategy)
            
            # Add time-based features to help XGBoost capture temporal patterns
            X_train_time, X_test_time = add_time_features(X_train, X_test)
            
            # Define model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = f"{strategy}_{timestamp}"
            
            # Split some training data for validation
            val_size = int(0.2 * len(X_train_time))
            X_val = X_train_time.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_final = X_train_time.iloc[:-val_size]
            y_train_final = y_train.iloc[:-val_size]
            
            # Train model
            model, importance_df = train_xgboost_model(
                X_train_final, y_train_final,
                X_val, y_val,
                model_name
            )
            
            # Evaluate model
            predictions_df, metrics = evaluate_model(
                model, 
                X_test_time, 
                y_test, 
                model_name
            )
            
            logger.info(f"\n{strategy} XGBoost model training and evaluation completed!\n")
            
            # Save summary info
            summary_file = EVAL_DIR / 'model_summary.csv'
            summary_data = {
                'model_name': model_name,
                'strategy': strategy,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'features': len(X_train.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
            }
            
            # Append to CSV if exists, otherwise create
            summary_df = pd.DataFrame([summary_data])
            if os.path.exists(summary_file):
                existing_df = pd.read_csv(summary_file)
                summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Model summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"ERROR processing {strategy} strategy: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nAll XGBoost model training and evaluation completed!")

if __name__ == "__main__":
    main()