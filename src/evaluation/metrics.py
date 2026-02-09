"""
Metrics for Time Series Forecasting Evaluation

Implements MAE, RMSE, MAPE, MASE, sMAPE as per forecasting best practices.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Note: MAPE is undefined when y_true = 0, so we add epsilon.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    sMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    
    sMAPE is more robust than MAPE when y_true is small.
    """
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(2 * np.abs(y_true - y_pred) / denominator)) * 100


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.
    
    MASE = MAE / mean(|y_train[t] - y_train[t-seasonality]|)
    
    MASE < 1 means the model beats naive forecast
    MASE > 1 means naive forecast is better
    
    Args:
        y_true: Test actuals
        y_pred: Predictions
        y_train: Training data (for scaling)
        seasonality: Seasonal period (1 for non-seasonal naive)
    """
    # Calculate naive forecast error on training data
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    
    if scale < 1e-10:
        return np.nan
    
    return mae(y_true, y_pred) / scale


def msse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Squared Scaled Error.
    
    Similar to MASE but using squared errors.
    """
    naive_errors = (y_train[seasonality:] - y_train[:-seasonality]) ** 2
    scale = np.mean(naive_errors)
    
    if scale < 1e-10:
        return np.nan
    
    mse = np.mean((y_true - y_pred) ** 2)
    return mse / scale


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Direction Accuracy (hit rate for direction of change).
    
    Measures how often the model correctly predicts up/down movement.
    """
    if len(y_true) < 2:
        return np.nan
    
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    return float(np.mean(true_direction == pred_direction)) * 100


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 1,
) -> Dict[str, float]:
    """
    Calculate all forecasting metrics.
    
    Args:
        y_true: Test actuals
        y_pred: Predictions
        y_train: Training data (for MASE)
        seasonality: Seasonal period
        
    Returns:
        Dictionary of metric names to values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    
    if y_train is not None and len(y_train) > seasonality:
        y_train = np.asarray(y_train)
        metrics["mase"] = mase(y_true, y_pred, y_train, seasonality)
        metrics["msse"] = msse(y_true, y_pred, y_train, seasonality)
    
    if len(y_true) > 1:
        metrics["direction_accuracy"] = direction_accuracy(y_true, y_pred)
    
    return metrics


def calculate_metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Optional[list] = None,
) -> pd.DataFrame:
    """
    Calculate metrics at different forecast horizons.
    
    Args:
        y_true: 2D array (n_samples, max_horizon) of actuals
        y_pred: 2D array (n_samples, max_horizon) of predictions
        horizons: List of horizons to evaluate (e.g., [1, 7, 30])
        
    Returns:
        DataFrame with metrics by horizon
    """
    if horizons is None:
        horizons = list(range(1, y_true.shape[1] + 1))
    
    results = []
    for h in horizons:
        if h > y_true.shape[1]:
            continue
        
        metrics = calculate_metrics(y_true[:, h-1], y_pred[:, h-1])
        metrics["horizon"] = h
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("horizon")


class MetricTracker:
    """Track metrics over time for monitoring."""
    
    def __init__(self):
        self.history = []
    
    def add(
        self,
        timestamp: str,
        commodity_id: str,
        metrics: Dict[str, float],
        horizon: int = 1,
    ) -> None:
        """Add a metrics observation."""
        self.history.append({
            "timestamp": timestamp,
            "commodity_id": commodity_id,
            "horizon": horizon,
            **metrics,
        })
    
    def get_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        return pd.DataFrame(self.history)
    
    def get_rolling_metrics(
        self,
        commodity_id: str,
        metric: str = "mae",
        window: int = 30,
    ) -> pd.Series:
        """Get rolling average of a metric."""
        df = self.get_history()
        df = df[df["commodity_id"] == commodity_id].sort_values("timestamp")
        return df[metric].rolling(window=window, min_periods=1).mean()
    
    def check_degradation(
        self,
        commodity_id: str,
        metric: str = "mae",
        threshold_pct: float = 0.2,
        window: int = 30,
    ) -> bool:
        """Check if metric has degraded beyond threshold."""
        rolling = self.get_rolling_metrics(commodity_id, metric, window)
        if len(rolling) < window * 2:
            return False
        
        recent = rolling.iloc[-window:].mean()
        earlier = rolling.iloc[-window*2:-window].mean()
        
        if earlier < 1e-10:
            return False
        
        degradation = (recent - earlier) / earlier
        return degradation > threshold_pct
