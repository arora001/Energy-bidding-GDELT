from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def compute_metrics(y_true, y_pred):
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