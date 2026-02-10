"""
Train/Test Split for Time Series Data

Implements chronological splitting with configurable ratios,
minimum test size, and optional gap to prevent data leakage.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class SplitResult:
    """Result of a train/test split."""
    
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_size: int
    test_size: int
    split_date: str
    gap_days: int
    
    def summary(self) -> str:
        """Human-readable summary."""
        train_start = self.train_df["date"].min()
        train_end = self.train_df["date"].max()
        test_start = self.test_df["date"].min()
        test_end = self.test_df["date"].max()
        return (
            f"Train: {train_start} -> {train_end} ({self.train_size} rows)\n"
            f"Test:  {test_start} -> {test_end} ({self.test_size} rows)\n"
            f"Gap:   {self.gap_days} days"
        )


class TimeSeriesSplitter:
    """
    Split time series data chronologically.
    
    Ensures no data leakage by respecting temporal order
    and optionally adding a gap between train and test sets.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        min_test_days: int = 30,
        gap_days: int = 0,
    ):
        """
        Initialize the splitter.
        
        Args:
            train_ratio: Fraction of data for training (0.0-1.0)
            min_test_days: Minimum number of rows in test set
            gap_days: Number of rows to skip between train and test
                     (prevents leakage from features like rolling windows)
        """
        if not 0.1 <= train_ratio <= 0.95:
            raise ValueError(f"train_ratio must be between 0.1 and 0.95, got {train_ratio}")
        
        self.train_ratio = train_ratio
        self.min_test_days = min_test_days
        self.gap_days = gap_days
    
    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> SplitResult:
        """
        Split DataFrame chronologically.
        
        Args:
            df: DataFrame to split (must be sorted by date)
            date_col: Name of date column
            
        Returns:
            SplitResult with train and test DataFrames
        """
        df = df.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        
        n = len(df)
        
        # Calculate split point
        train_end_idx = int(n * self.train_ratio)
        
        # Ensure minimum test size
        max_train = n - self.min_test_days - self.gap_days
        if train_end_idx > max_train:
            train_end_idx = max(max_train, self.min_test_days)
            logger.warning(
                f"Adjusted train size to {train_end_idx} to ensure "
                f"min {self.min_test_days} test rows"
            )
        
        # Apply gap
        test_start_idx = train_end_idx + self.gap_days
        
        train_df = df.iloc[:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:].copy()
        
        split_date = str(df.iloc[train_end_idx - 1][date_col])
        
        logger.info(
            f"Split: train={len(train_df)} rows, test={len(test_df)} rows, "
            f"gap={self.gap_days}, split_date={split_date}"
        )
        
        return SplitResult(
            train_df=train_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
            train_size=len(train_df),
            test_size=len(test_df),
            split_date=split_date,
            gap_days=self.gap_days,
        )
    
    def split_features_target(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "close",
        date_col: str = "date",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split into X_train, y_train, X_test, y_test.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            date_col: Name of date column
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        result = self.split(df, date_col)
        
        X_train = result.train_df[feature_cols]
        y_train = result.train_df[target_col]
        X_test = result.test_df[feature_cols]
        y_test = result.test_df[target_col]
        
        return X_train, y_train, X_test, y_test


def train_test_split_ts(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    min_test_days: int = 30,
    gap_days: int = 0,
    date_col: str = "date",
) -> SplitResult:
    """
    Convenience function for chronological train/test split.
    
    Args:
        df: DataFrame to split
        train_ratio: Fraction for training
        min_test_days: Minimum test set size
        gap_days: Gap between train and test
        date_col: Date column name
        
    Returns:
        SplitResult with train and test DataFrames
    """
    splitter = TimeSeriesSplitter(
        train_ratio=train_ratio,
        min_test_days=min_test_days,
        gap_days=gap_days,
    )
    return splitter.split(df, date_col)
