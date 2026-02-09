"""
Data Cleaner for Commodity Price Data

Handles frequency alignment, missing value treatment, 
and data cleaning for the Silver layer.
"""

from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    """Clean and align commodity price data."""
    
    def __init__(
        self,
        frequency: Literal["calendar", "business"] = "business",
        fill_method: Literal["ffill", "none", "interpolate"] = "ffill",
        max_gap_days: int = 5,
    ):
        """
        Initialize the data cleaner.
        
        Args:
            frequency: Alignment frequency - 'calendar' for all days,
                      'business' for business days only
            fill_method: How to handle missing values:
                        - 'ffill': forward fill (recommended)
                        - 'none': keep as NaN
                        - 'interpolate': linear interpolation
            max_gap_days: Maximum consecutive days to fill (beyond this, keep NaN)
        """
        self.frequency = frequency
        self.fill_method = fill_method
        self.max_gap_days = max_gap_days
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process the data.
        
        Args:
            df: DataFrame with columns (date, close, commodity_id)
            
        Returns:
            Cleaned DataFrame with additional metadata columns
        """
        df = df.copy()
        
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # Process each commodity separately
        if "commodity_id" in df.columns:
            commodities = df["commodity_id"].unique()
            cleaned_dfs = []
            
            for commodity_id in commodities:
                commodity_df = df[df["commodity_id"] == commodity_id].copy()
                cleaned = self._clean_single_commodity(commodity_df)
                cleaned_dfs.append(cleaned)
            
            return pd.concat(cleaned_dfs, ignore_index=True)
        else:
            return self._clean_single_commodity(df)
    
    def _clean_single_commodity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for a single commodity."""
        df = df.copy()
        
        # Step 1: Remove duplicates (keep last)
        df = df.drop_duplicates(subset=["date"], keep="last")
        
        # Step 2: Align to target frequency
        df = self._align_frequency(df)
        
        # Step 3: Add metadata columns
        df = self._add_metadata(df)
        
        # Step 4: Handle missing values
        df = self._handle_missing(df)
        
        # Step 5: Remove extreme outliers (optional, only flag them)
        df = self._flag_outliers(df)
        
        return df
    
    def _align_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align data to target frequency."""
        df = df.set_index("date")
        
        # Create target date range
        start_date = df.index.min()
        end_date = df.index.max()
        
        if self.frequency == "business":
            target_index = pd.bdate_range(start=start_date, end=end_date)
        else:
            target_index = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Reindex to target frequency
        df = df.reindex(target_index)
        df.index.name = "date"
        
        return df.reset_index()
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns for tracking data quality."""
        # Flag original vs imputed values
        df["is_original"] = df["close"].notna()
        
        # Count consecutive missing before this point
        df["missing_streak"] = (
            (~df["is_original"])
            .groupby((df["is_original"]).cumsum())
            .cumsum()
        )
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        if self.fill_method == "none":
            return df
        
        # Identify gaps that are too large
        large_gaps = df["missing_streak"] > self.max_gap_days
        
        if self.fill_method == "ffill":
            df["close"] = df["close"].ffill()
        elif self.fill_method == "interpolate":
            df["close"] = df["close"].interpolate(method="linear")
        
        # Reset values in large gaps to NaN
        df.loc[large_gaps, "close"] = np.nan
        
        # Update is_original flag
        df["is_imputed"] = ~df["is_original"] & df["close"].notna()
        
        return df
    
    def _flag_outliers(self, df: pd.DataFrame, window: int = 30, threshold: float = 4.0) -> pd.DataFrame:
        """Flag potential outliers using rolling z-score."""
        if len(df) < window:
            df["is_outlier"] = False
            return df
        
        close = df["close"]
        rolling_mean = close.rolling(window=window, min_periods=10, center=True).mean()
        rolling_std = close.rolling(window=window, min_periods=10, center=True).std()
        
        z_score = np.abs((close - rolling_mean) / rolling_std)
        df["is_outlier"] = z_score > threshold
        df["z_score"] = z_score
        
        return df
    
    def get_quality_summary(self, df: pd.DataFrame) -> dict:
        """Generate a quality summary for the cleaned data."""
        total_rows = len(df)
        original_rows = df["is_original"].sum() if "is_original" in df.columns else total_rows
        imputed_rows = df["is_imputed"].sum() if "is_imputed" in df.columns else 0
        missing_rows = df["close"].isna().sum()
        outlier_rows = df["is_outlier"].sum() if "is_outlier" in df.columns else 0
        
        return {
            "total_rows": int(total_rows),
            "original_rows": int(original_rows),
            "imputed_rows": int(imputed_rows),
            "missing_rows": int(missing_rows),
            "outlier_rows": int(outlier_rows),
            "original_pct": float(original_rows / total_rows) if total_rows > 0 else 0,
            "imputed_pct": float(imputed_rows / total_rows) if total_rows > 0 else 0,
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
        }


def clean_data(
    df: pd.DataFrame,
    frequency: Literal["calendar", "business"] = "business",
    fill_method: Literal["ffill", "none", "interpolate"] = "ffill",
    max_gap_days: int = 5,
) -> pd.DataFrame:
    """
    Convenience function to clean data.
    
    Args:
        df: DataFrame to clean
        frequency: Alignment frequency
        fill_method: Missing value handling method
        max_gap_days: Maximum gap to fill
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(
        frequency=frequency,
        fill_method=fill_method,
        max_gap_days=max_gap_days,
    )
    return cleaner.clean(df)
