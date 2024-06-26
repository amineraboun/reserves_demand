import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error
)

def evaluate_metrics(y_true, y_pred):
    try:
        msle = mean_squared_log_error(y_true, y_pred)
    except:
        msle = np.nan
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R2': r2_score(y_true, y_pred),
        'MSLE': msle,
        'MedianAE': median_absolute_error(y_true, y_pred)
    }
    return pd.Series(metrics)

def mis(y, yqlb, yqub, Q):
    return ((yqub - yqlb) + (2 / Q) * (yqlb - y) * (y < yqlb) + (2 / Q) * (y - yqub) * (y > yqub)).mean()

def perf_metrics(y, ypred, yqlb, yqub, q):
    """
    Computes performance metrics for the fitted curve.
    The performance metrics include 
        RMSE: Root Mean Squared Error
        MAE: Mean Absolute Error
        MAPE: Mean Absolute Percentage Error
        R2: R-squared
        MSLE: Mean Squared Log Error
        MedianAE: Median Absolute Error
        MIS: Mean Interval Score
    """
    metrics = evaluate_metrics(y, ypred)
    metrics['MIS'] = mis(y, yqlb, yqub, q)
    return metrics