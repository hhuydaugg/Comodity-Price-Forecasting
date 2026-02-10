"""
Feature Generator for Commodity Price Forecasting

Creates lag features, rolling features, calendar features, 
and volatility proxies for time series ML models.
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureGenerator:
    """Generate features for time series forecasting."""
    
    def __init__(
        self,
        lag_days: List[int] = None,
        rolling_windows: List[int] = None,
        rolling_stats: List[str] = None,
        calendar_features: bool = True,
        volatility_features: bool = True,
        target_column: str = "close",
    ):
        """
        Initialize the feature generator.
        
        Args:
            lag_days: List of lag periods (e.g., [1, 2, 7, 14, 30])
            rolling_windows: List of rolling window sizes (e.g., [7, 14, 30])
            rolling_stats: List of stats to compute (e.g., ['mean', 'std', 'min', 'max'])
            calendar_features: Whether to add calendar-based features
            volatility_features: Whether to add volatility-based features
            target_column: Name of the column to generate features from
        """
        self.lag_days = lag_days or [1, 2, 3, 5, 7, 14, 21, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30, 60]
        self.rolling_stats = rolling_stats or ["mean", "std", "min", "max"]
        self.calendar_features = calendar_features
        self.volatility_features = volatility_features
        self.target_column = target_column
        
        self.feature_names_: List[str] = []
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features.
        
        Args:
            df: DataFrame with at least 'date' and target column
            
        Returns:
            DataFrame with all generated features
        """
        df = df.copy()
        
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # Check if we need to process by commodity
        if "commodity_id" in df.columns:
            commodities = df["commodity_id"].unique()
            feature_dfs = []
            
            for commodity_id in commodities:
                commodity_df = df[df["commodity_id"] == commodity_id].copy()
                featured = self._generate_features(commodity_df)
                feature_dfs.append(featured)
            
            result = pd.concat(feature_dfs, ignore_index=True)
        else:
            result = self._generate_features(df)
        
        self.feature_names_ = self._get_feature_names(result)
        return result
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for a single commodity."""
        target = self.target_column
        
        # 1. Lag features
        df = self._add_lag_features(df, target)
        
        # 2. Rolling features
        df = self._add_rolling_features(df, target)
        
        # 3. Calendar features
        if self.calendar_features:
            df = self._add_calendar_features(df)
        
        # 4. Volatility features
        if self.volatility_features:
            df = self._add_volatility_features(df, target)
        
        # 5. Return-based features
        df = self._add_return_features(df, target)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add lag features."""
        for lag in self.lag_days:
            df[f"lag_{lag}"] = df[target].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add rolling window features."""
        for window in self.rolling_windows:
            for stat in self.rolling_stats:
                col_name = f"rolling_{stat}_{window}"
                
                if stat == "mean":
                    df[col_name] = df[target].shift(1).rolling(window=window, min_periods=1).mean()
                elif stat == "std":
                    df[col_name] = df[target].shift(1).rolling(window=window, min_periods=2).std()
                elif stat == "min":
                    df[col_name] = df[target].shift(1).rolling(window=window, min_periods=1).min()
                elif stat == "max":
                    df[col_name] = df[target].shift(1).rolling(window=window, min_periods=1).max()
                elif stat == "median":
                    df[col_name] = df[target].shift(1).rolling(window=window, min_periods=1).median()
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        date = df["date"]
        
        # Day of week (0 = Monday, 6 = Sunday)
        df["day_of_week"] = date.dt.dayofweek
        
        # One-hot encode day of week
        for i in range(5):  # Only weekdays (0-4)
            df[f"is_dow_{i}"] = (date.dt.dayofweek == i).astype(int)
        
        # Month (1-12)
        df["month"] = date.dt.month
        
        # Week of year (1-52)
        df["week_of_year"] = date.dt.isocalendar().week.astype(int)
        
        # Day of month
        df["day_of_month"] = date.dt.day
        
        # Is month start/end
        df["is_month_start"] = date.dt.is_month_start.astype(int)
        df["is_month_end"] = date.dt.is_month_end.astype(int)
        
        # Is quarter start/end
        df["is_quarter_start"] = date.dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = date.dt.is_quarter_end.astype(int)
        
        # Year
        df["year"] = date.dt.year
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add volatility-based features."""
        # Check if we have enough data
        if len(df) < 2:
            logger.warning(f"Not enough data for volatility features: {len(df)} rows")
            # Add empty columns to maintain schema
            for window in [7, 14, 30]:
                df[f"volatility_{window}"] = np.nan
            df["volatility_ewm_14"] = np.nan
            for window in [7, 14]:
                df[f"range_pct_{window}"] = np.nan
            return df
        
        # Calculate returns for volatility
        returns = df[target].pct_change(fill_method=None)
        
        # Check if returns has valid data
        if returns.notna().sum() < 2:
            logger.warning(f"Not enough valid returns for volatility calculation")
            for window in [7, 14, 30]:
                df[f"volatility_{window}"] = np.nan
            df["volatility_ewm_14"] = np.nan
            for window in [7, 14]:
                df[f"range_pct_{window}"] = np.nan
            return df
        
        # Rolling volatility (std of returns)
        for window in [7, 14, 30]:
            df[f"volatility_{window}"] = returns.shift(1).rolling(window=window, min_periods=2).std()
        
        # EWMA volatility
        df["volatility_ewm_14"] = returns.shift(1).ewm(span=14, min_periods=7).std()
        
        # High-Low range proxy (if we only have close, use rolling range)
        for window in [7, 14]:
            rolling_max = df[target].shift(1).rolling(window=window, min_periods=1).max()
            rolling_min = df[target].shift(1).rolling(window=window, min_periods=1).min()
            df[f"range_pct_{window}"] = (rolling_max - rolling_min) / rolling_min
        
        return df
    
    
    def _add_return_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Add return-based features."""
        # Check if we have enough data
        if len(df) < 2:
            logger.warning(f"Not enough data for return features: {len(df)} rows")
            # Add empty columns to maintain schema
            df["return_1d"] = np.nan
            df["return_5d"] = np.nan
            df["return_20d"] = np.nan
            df["return_1d_lag1"] = np.nan
            df["return_5d_lag1"] = np.nan
            df["momentum_5d"] = np.nan
            df["momentum_20d"] = np.nan
            df["consecutive_direction_lag1"] = np.nan
            return df
        
        # Check if target column has valid data
        if df[target].notna().sum() < 2:
            logger.warning(f"Not enough valid data in target column for return calculation")
            df["return_1d"] = np.nan
            df["return_5d"] = np.nan
            df["return_20d"] = np.nan
            df["return_1d_lag1"] = np.nan
            df["return_5d_lag1"] = np.nan
            df["momentum_5d"] = np.nan
            df["momentum_20d"] = np.nan
            df["consecutive_direction_lag1"] = np.nan
            return df
        
        # Simple returns
        df["return_1d"] = df[target].pct_change(1, fill_method=None)
        df["return_5d"] = df[target].pct_change(5, fill_method=None)
        df["return_20d"] = df[target].pct_change(20, fill_method=None)
        
        # Lagged returns (for features, not target)
        df["return_1d_lag1"] = df["return_1d"].shift(1)
        df["return_5d_lag1"] = df["return_5d"].shift(1)
        
        # Momentum (sign of recent returns)
        df["momentum_5d"] = np.sign(df["return_5d"].shift(1))
        df["momentum_20d"] = np.sign(df["return_20d"].shift(1))
        
        # Consecutive up/down days
        df["return_sign"] = np.sign(df["return_1d"])
        df["consecutive_direction"] = (
            df["return_sign"]
            .groupby((df["return_sign"] != df["return_sign"].shift()).cumsum())
            .cumcount() + 1
        ) * df["return_sign"]
        df["consecutive_direction_lag1"] = df["consecutive_direction"].shift(1)
        
        # Clean up intermediate columns
        df = df.drop(columns=["return_sign"])
        
        return df
    
    def _get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of generated feature column names."""
        exclude_cols = ["date", "close", "commodity_id", "target", 
                       "is_original", "is_imputed", "is_outlier", "z_score", 
                       "missing_streak", "consecutive_direction"]
        return [c for c in df.columns if c not in exclude_cols]
    
    def get_feature_names(self) -> List[str]:
        """Return the list of feature names after generation."""
        return self.feature_names_.copy()
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
        drop_na: bool = True,
    ) -> tuple:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Tuple of (X DataFrame, y Series, valid indices)
        """
        feature_cols = [c for c in self.feature_names_ 
                       if c not in ["date", target_column, "commodity_id"]]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy() if target_column in df.columns else None
        
        if drop_na:
            if y is not None:
                valid_mask = X.notna().all(axis=1) & y.notna()
            else:
                valid_mask = X.notna().all(axis=1)
            
            X = X[valid_mask]
            y = y[valid_mask] if y is not None else None
            valid_indices = df.index[valid_mask]
        else:
            valid_indices = df.index
        
        return X, y, valid_indices


def generate_features(
    df: pd.DataFrame,
    lag_days: List[int] = None,
    rolling_windows: List[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to generate features.
    
    Args:
        df: DataFrame with 'date' and 'close' columns
        lag_days: List of lag periods
        rolling_windows: List of rolling window sizes
        **kwargs: Additional arguments for FeatureGenerator
        
    Returns:
        DataFrame with generated features
    """
    generator = FeatureGenerator(
        lag_days=lag_days,
        rolling_windows=rolling_windows,
        **kwargs,
    )
    return generator.generate(df)
