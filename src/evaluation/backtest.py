"""
Backtesting Framework for Time Series Forecasting

Implements walk-forward and rolling-origin validation
as per time series best practices.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import calculate_metrics


@dataclass
class BacktestFold:
    """Represents a single fold in backtesting."""
    
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int


@dataclass
class BacktestResult:
    """Results from a single backtest fold."""
    
    fold_id: int
    model_name: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None


@dataclass  
class BacktestSummary:
    """Aggregated results across all folds."""
    
    model_name: str
    commodity_id: str
    n_folds: int
    results: List[BacktestResult] = field(default_factory=list)
    
    def get_mean_metrics(self) -> Dict[str, float]:
        """Get mean metrics across folds."""
        if not self.results:
            return {}
        
        all_metrics = [r.metrics for r in self.results]
        metric_names = all_metrics[0].keys()
        
        return {
            name: np.mean([m.get(name, np.nan) for m in all_metrics])
            for name in metric_names
        }
    
    def get_std_metrics(self) -> Dict[str, float]:
        """Get std of metrics across folds."""
        if not self.results:
            return {}
        
        all_metrics = [r.metrics for r in self.results]
        metric_names = all_metrics[0].keys()
        
        return {
            name: np.std([m.get(name, np.nan) for m in all_metrics])
            for name in metric_names
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in self.results:
            row = {
                "fold_id": r.fold_id,
                "model_name": r.model_name,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                **r.metrics,
            }
            rows.append(row)
        return pd.DataFrame(rows)


class TimeSeriesBacktest:
    """
    Walk-forward backtesting for time series.
    
    Implements three strategies:
    - walk_forward: Expanding window, fixed test size
    - rolling: Fixed window, slides forward
    - expanding: Expanding window, single test at end
    """
    
    def __init__(
        self,
        method: Literal["walk_forward", "rolling", "expanding"] = "walk_forward",
        initial_train_size: int = 504,  # 2 years
        test_size: int = 30,  # 1 month
        step_size: int = 30,  # Step forward 1 month
        n_folds: Optional[int] = None,  # Limit number of folds
    ):
        """
        Initialize backtester.
        
        Args:
            method: Backtesting method
            initial_train_size: Initial training window size
            test_size: Size of each test fold
            step_size: How much to move forward each fold
            n_folds: Maximum number of folds (None for all)
        """
        self.method = method
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.n_folds = n_folds
    
    def get_folds(self, df: pd.DataFrame) -> Generator[BacktestFold, None, None]:
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with 'date' column
            
        Yields:
            BacktestFold objects
        """
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        
        if n < self.initial_train_size + self.test_size:
            raise ValueError(
                f"Not enough data: {n} rows, need {self.initial_train_size + self.test_size}"
            )
        
        fold_id = 0
        train_end_idx = self.initial_train_size - 1
        
        while train_end_idx + self.test_size < n:
            train_start_idx = 0 if self.method != "rolling" else max(0, train_end_idx - self.initial_train_size + 1)
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_size - 1, n - 1)
            
            yield BacktestFold(
                fold_id=fold_id,
                train_start=df.iloc[train_start_idx]["date"],
                train_end=df.iloc[train_end_idx]["date"],
                test_start=df.iloc[test_start_idx]["date"],
                test_end=df.iloc[test_end_idx]["date"],
                train_size=train_end_idx - train_start_idx + 1,
                test_size=test_end_idx - test_start_idx + 1,
            )
            
            fold_id += 1
            train_end_idx += self.step_size
            
            if self.n_folds and fold_id >= self.n_folds:
                break
    
    def split_data(
        self,
        df: pd.DataFrame,
        fold: BacktestFold,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data according to a fold.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.sort_values("date")
        
        train = df[(df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)]
        test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
        
        return train.reset_index(drop=True), test.reset_index(drop=True)
    
    def run(
        self,
        df: pd.DataFrame,
        model,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
        commodity_id: str = "unknown",
    ) -> BacktestSummary:
        """
        Run backtest for a single model.
        
        Args:
            df: Full dataset with features
            model: Model instance with fit() and predict() methods
            target_col: Name of target column
            feature_cols: List of feature columns (for ML models)
            commodity_id: ID for reporting
            
        Returns:
            BacktestSummary with all fold results
        """
        summary = BacktestSummary(
            model_name=model.name,
            commodity_id=commodity_id,
            n_folds=0,
        )
        
        for fold in self.get_folds(df):
            train_df, test_df = self.split_data(df, fold)
            
            try:
                result = self._evaluate_fold(
                    model=model,
                    train_df=train_df,
                    test_df=test_df,
                    fold=fold,
                    target_col=target_col,
                    feature_cols=feature_cols,
                )
                summary.results.append(result)
                summary.n_folds += 1
                
            except Exception as e:
                logger.warning(f"Fold {fold.fold_id} failed: {e}")
        
        return summary
    
    def _evaluate_fold(
        self,
        model,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold: BacktestFold,
        target_col: str,
        feature_cols: Optional[List[str]],
    ) -> BacktestResult:
        """Evaluate a single fold."""
        y_train = train_df[target_col]
        y_test = test_df[target_col].values
        
        # Fit model
        if feature_cols:
            # ML model with features
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            model.fit(y_train, X_train)
            predictions = model.predict(horizon=len(y_test), X=X_test)
        else:
            # Statistical/baseline model
            model.fit(y_train)
            predictions = model.predict(horizon=len(y_test))
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=predictions,
            y_train=y_train.values,
        )
        
        return BacktestResult(
            fold_id=fold.fold_id,
            model_name=model.name,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            metrics=metrics,
            predictions=predictions,
            actuals=y_test,
        )


def compare_models(
    df: pd.DataFrame,
    models: list,
    target_col: str = "close",
    feature_cols: Optional[List[str]] = None,
    commodity_id: str = "unknown",
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Compare multiple models using backtesting.
    
    Args:
        df: Dataset with target and features
        models: List of model instances
        target_col: Target column name
        feature_cols: Feature columns for ML models
        commodity_id: ID for reporting
        **backtest_kwargs: Arguments for TimeSeriesBacktest
        
    Returns:
        DataFrame comparing all models
    """
    backtester = TimeSeriesBacktest(**backtest_kwargs)
    
    results = []
    for model in models:
        try:
            summary = backtester.run(
                df=df,
                model=model,
                target_col=target_col,
                feature_cols=feature_cols,
                commodity_id=commodity_id,
            )
            
            mean_metrics = summary.get_mean_metrics()
            std_metrics = summary.get_std_metrics()
            
            row = {
                "model": model.name,
                "commodity_id": commodity_id,
                "n_folds": summary.n_folds,
            }
            for metric, value in mean_metrics.items():
                row[f"{metric}_mean"] = value
                row[f"{metric}_std"] = std_metrics.get(metric, np.nan)
            
            results.append(row)
            
        except Exception as e:
            logger.error(f"Model {model.name} failed: {e}")
    
    return pd.DataFrame(results)
