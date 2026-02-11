"""
Future Price Forecaster

Iterative multi-step forecasting: predicts N days into the future
by appending each prediction and recalculating features.
"""

from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.ingestion.loader import CommodityLoader
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.transformer import TargetTransformer
from src.features.generator import FeatureGenerator


class FutureForecaster:
    """
    Predict commodity prices N days into the future.
    
    Uses iterative one-step-ahead forecasting:
    1. Start with the latest available data + features
    2. Predict the next day's price
    3. Append predicted price back into the dataset
    4. Recalculate features for the new row
    5. Repeat for N days
    """
    
    def __init__(
        self,
        loader: Optional[CommodityLoader] = None,
        cleaner: Optional[DataCleaner] = None,
        feature_generator: Optional[FeatureGenerator] = None,
        transformer: Optional[TargetTransformer] = None,
    ):
        self.loader = loader or CommodityLoader()
        self.cleaner = cleaner or DataCleaner(frequency="auto")
        self.feature_generator = feature_generator or FeatureGenerator(
            lag_days=[1, 2, 3, 5, 7, 14, 21, 30],
            rolling_windows=[7, 14, 30, 60],
        )
        self.transformer = transformer or TargetTransformer(method="none")
    
    def prepare_base_data(self, commodity_id: str) -> pd.DataFrame:
        """Load and prepare historical data as the base for forecasting."""
        df = self.loader.load_commodity(commodity_id)
        df = self.cleaner.clean(df)
        df = self.transformer.transform(df)
        df = self.feature_generator.generate(df)
        return df
    
    def forecast(
        self,
        model,
        commodity_id: str,
        n_days: int = 30,
        feature_cols: Optional[list] = None,
        confidence_std: float = 1.96,
        residual_std: Optional[float] = None,
        base_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Forecast N days into the future using iterative prediction.
        
        Args:
            model: A fitted MLModel instance (must have .predict(X=...))
            commodity_id: Commodity identifier
            n_days: Number of days to forecast
            feature_cols: Feature columns to use (auto-detected if None)
            confidence_std: Z-score for confidence interval (1.96 = 95%)
            residual_std: Standard deviation of residuals for CI
                         (if None, uses 2% of last price as rough estimate)
            base_df: Pre-prepared base DataFrame (skips load/clean/features)
            
        Returns:
            DataFrame with columns:
            - date: forecast date
            - predicted_price: point forecast
            - ci_lower: confidence interval lower bound
            - ci_upper: confidence interval upper bound
            - day_ahead: number of days ahead (1, 2, ..., N)
        """
        # Prepare base data
        if base_df is not None:
            df = base_df.copy()
        else:
            df = self.prepare_base_data(commodity_id)
        
        # Auto-detect feature columns
        if feature_cols is None:
            feature_cols = [c for c in self.feature_generator.get_feature_names()
                          if c in df.columns]
        
        # Drop NaN rows to get clean base
        df_clean = df.dropna(subset=feature_cols + ["close"]).copy()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data for forecasting")
        
        last_date = pd.Timestamp(df_clean["date"].max())
        last_price = float(df_clean["close"].iloc[-1])
        
        # Estimate residual std if not provided
        if residual_std is None:
            residual_std = last_price * 0.02  # rough 2% estimate
        
        logger.info(
            f"Starting {n_days}-day forecast for {commodity_id} "
            f"from {last_date.date()} (last price: {last_price:.2f})"
        )
        
        # Working copy - keep enough history for feature calculation
        work_df = df_clean.copy()
        forecasts = []
        current_date = last_date  # track date incrementally
        
        for day in range(1, n_days + 1):
            # Advance to next business day
            current_date = current_date + timedelta(days=1)
            current_date = self._next_business_day(current_date)
            
            # Get features
            if hasattr(model, "seq_len"):
                # For transformers, get the sequence window
                # Maximize available history up to seq_len
                sl = getattr(model, "seq_len") 
                X_latest = work_df[feature_cols].iloc[-sl:]
            else:
                # For ML models, get only the last row
                X_latest = work_df[feature_cols].iloc[[-1]]
            
            # Check for NaN in features
            if X_latest.isna().any().any():
                # nan_cols = X_latest.columns[X_latest.isna().any()].tolist()
                # logger.warning(f"Day {day}: NaN in features, filling with 0")
                X_latest = X_latest.fillna(0)
            
            # Predict
            try:
                if hasattr(model, "predict_single"):
                    pred = model.predict_single(X_latest)
                else:
                    pred = float(model.predict(X=X_latest)[0])
            except Exception as e:
                logger.error(f"Prediction failed at day {day}: {e}")
                break
            
            # Ensure prediction is positive
            pred = max(pred, 0.01)
            
            # Growing confidence interval (uncertainty increases with horizon)
            ci_width = confidence_std * residual_std * np.sqrt(day)
            
            forecasts.append({
                "date": current_date,
                "predicted_price": pred,
                "ci_lower": max(pred - ci_width, 0),
                "ci_upper": pred + ci_width,
                "day_ahead": day,
            })
            
            # Append prediction to working data for next iteration
            new_row = work_df.iloc[-1:].copy()
            new_row["date"] = current_date
            new_row["close"] = pred
            if "target" in new_row.columns:
                new_row["target"] = pred
            if "is_original" in new_row.columns:
                new_row["is_original"] = False
            if "is_imputed" in new_row.columns:
                new_row["is_imputed"] = False
            
            work_df = pd.concat([work_df, new_row], ignore_index=True)
            
            # Recalculate features for the new row
            work_df = self._recalculate_features(work_df, feature_cols)
        
        result = pd.DataFrame(forecasts)
        
        logger.info(
            f"Forecast complete: {len(result)} days, "
            f"last predicted price: {result['predicted_price'].iloc[-1]:.2f} "
            f"({((result['predicted_price'].iloc[-1] / last_price) - 1) * 100:+.2f}%)"
        )
        
        return result
    
    def _next_business_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get the next business day (skip weekends)."""
        while date.weekday() >= 5:  # Saturday=5, Sunday=6
            date += timedelta(days=1)
        return date
    
    def _recalculate_features(
        self, df: pd.DataFrame, feature_cols: list
    ) -> pd.DataFrame:
        """
        Recalculate lag and rolling features for the last row.
        
        Instead of regenerating all features (expensive), we update
        only the last row's features using simple shift/rolling logic.
        """
        target = "close"
        n = len(df)
        last_idx = n - 1
        
        # Update lag features
        for col in feature_cols:
            if col.startswith("lag_"):
                try:
                    lag = int(col.split("_")[1])
                    if last_idx >= lag:
                        df.loc[df.index[last_idx], col] = df[target].iloc[last_idx - lag]
                except (ValueError, IndexError):
                    pass
            
            elif col.startswith("rolling_"):
                # rolling_mean_7, rolling_std_14, etc.
                parts = col.split("_")
                if len(parts) >= 3:
                    try:
                        stat = parts[1]
                        window = int(parts[2])
                        # Use shifted data (exclude current row)
                        start = max(0, last_idx - window)
                        values = df[target].iloc[start:last_idx].values
                        
                        if len(values) > 0:
                            if stat == "mean":
                                df.loc[df.index[last_idx], col] = np.mean(values)
                            elif stat == "std":
                                df.loc[df.index[last_idx], col] = np.std(values, ddof=1) if len(values) > 1 else 0
                            elif stat == "min":
                                df.loc[df.index[last_idx], col] = np.min(values)
                            elif stat == "max":
                                df.loc[df.index[last_idx], col] = np.max(values)
                    except (ValueError, IndexError):
                        pass
            
            elif col.startswith("return_") and not col.endswith("_lag1"):
                # return_1d, return_5d, return_20d
                try:
                    period = int(col.split("_")[1].replace("d", ""))
                    if last_idx >= period:
                        prev = df[target].iloc[last_idx - period]
                        curr = df[target].iloc[last_idx]
                        if prev != 0:
                            df.loc[df.index[last_idx], col] = (curr - prev) / prev
                except (ValueError, IndexError):
                    pass
            
            elif col == "return_1d_lag1":
                if last_idx >= 2:
                    prev2 = df[target].iloc[last_idx - 2]
                    prev1 = df[target].iloc[last_idx - 1]
                    if prev2 != 0:
                        df.loc[df.index[last_idx], col] = (prev1 - prev2) / prev2
            
            elif col.startswith("volatility_") and not col.startswith("volatility_ewm"):
                try:
                    window = int(col.split("_")[1])
                    start = max(0, last_idx - window - 1)
                    values = df[target].iloc[start:last_idx].values
                    if len(values) > 1:
                        returns = np.diff(values) / values[:-1]
                        df.loc[df.index[last_idx], col] = np.std(returns, ddof=1)
                except (ValueError, IndexError):
                    pass
        
        return df


def forecast_future(
    model,
    commodity_id: str,
    n_days: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function for future price forecasting.
    
    Args:
        model: Fitted model instance
        commodity_id: Commodity ID
        n_days: Days to forecast
        
    Returns:
        DataFrame with date, predicted_price, ci_lower, ci_upper, day_ahead
    """
    forecaster = FutureForecaster()
    return forecaster.forecast(model, commodity_id, n_days, **kwargs)
