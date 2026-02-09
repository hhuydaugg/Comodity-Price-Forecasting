"""
Target Transformer for Commodity Price Data

Handles transformations between price levels and returns/log-returns
for better model training and predictions.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class TargetTransformer:
    """Transform target variable between price and returns."""
    
    def __init__(
        self,
        method: Literal["none", "log", "return", "log_return"] = "none",
        clip_returns: Optional[float] = 0.5,  # Clip extreme returns at 50%
    ):
        """
        Initialize the transformer.
        
        Args:
            method: Transformation method:
                   - 'none': Use raw price
                   - 'log': Log transform (log(price))
                   - 'return': Simple return ((p_t - p_{t-1}) / p_{t-1})
                   - 'log_return': Log return (log(p_t / p_{t-1}))
            clip_returns: Clip returns at this absolute value (e.g., 0.5 = ±50%)
        """
        self.method = method
        self.clip_returns = clip_returns
        
        # Store reference values for inverse transform
        self._reference_prices: dict = {}
    
    def fit(self, df: pd.DataFrame, commodity_id: str = "default") -> "TargetTransformer":
        """
        Fit the transformer (store reference values).
        
        Args:
            df: DataFrame with 'close' column
            commodity_id: ID for storing reference
            
        Returns:
            self for chaining
        """
        if self.method in ["return", "log_return"]:
            # Store last price for inverse transform
            self._reference_prices[commodity_id] = df["close"].iloc[-1]
        elif self.method == "log":
            # Store any reference if needed
            pass
        
        return self
    
    def transform(self, df: pd.DataFrame, commodity_id: str = "default") -> pd.DataFrame:
        """
        Transform the target variable.
        
        Args:
            df: DataFrame with 'close' column
            commodity_id: ID for reference lookup
            
        Returns:
            DataFrame with transformed 'target' column
        """
        df = df.copy()
        close = df["close"].copy()
        
        if self.method == "none":
            df["target"] = close
            
        elif self.method == "log":
            # Log transform (handle zeros/negatives)
            df["target"] = np.log(close.clip(lower=1e-8))
            
        elif self.method == "return":
            # Simple return
            returns = close.pct_change()
            if self.clip_returns:
                returns = returns.clip(-self.clip_returns, self.clip_returns)
            df["target"] = returns
            
        elif self.method == "log_return":
            # Log return
            log_returns = np.log(close / close.shift(1))
            if self.clip_returns:
                log_returns = log_returns.clip(-self.clip_returns, self.clip_returns)
            df["target"] = log_returns
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return df
    
    def inverse_transform(
        self,
        predictions: np.ndarray,
        reference_price: Optional[float] = None,
        commodity_id: str = "default",
    ) -> np.ndarray:
        """
        Inverse transform predictions back to price.
        
        Args:
            predictions: Array of predictions in transformed space
            reference_price: Reference price for return methods
                           (if None, uses stored reference)
            commodity_id: ID for reference lookup
            
        Returns:
            Array of predictions in price space
        """
        predictions = np.asarray(predictions)
        
        if self.method == "none":
            return predictions
        
        elif self.method == "log":
            return np.exp(predictions)
        
        elif self.method == "return":
            if reference_price is None:
                reference_price = self._reference_prices.get(commodity_id)
            if reference_price is None:
                raise ValueError("Reference price required for return inverse transform")
            
            # p_t = p_{t-1} * (1 + r_t)
            prices = np.zeros_like(predictions)
            prices[0] = reference_price * (1 + predictions[0])
            for i in range(1, len(predictions)):
                prices[i] = prices[i-1] * (1 + predictions[i])
            return prices
        
        elif self.method == "log_return":
            if reference_price is None:
                reference_price = self._reference_prices.get(commodity_id)
            if reference_price is None:
                raise ValueError("Reference price required for log_return inverse transform")
            
            # p_t = p_{t-1} * exp(log_r_t)
            prices = np.zeros_like(predictions)
            prices[0] = reference_price * np.exp(predictions[0])
            for i in range(1, len(predictions)):
                prices[i] = prices[i-1] * np.exp(predictions[i])
            return prices
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def inverse_transform_single(
        self,
        prediction: float,
        reference_price: float,
    ) -> float:
        """
        Inverse transform a single prediction.
        
        Args:
            prediction: Single prediction value
            reference_price: The price at time t to predict t+1
            
        Returns:
            Predicted price
        """
        if self.method == "none":
            return prediction
        
        elif self.method == "log":
            return np.exp(prediction)
        
        elif self.method == "return":
            return reference_price * (1 + prediction)
        
        elif self.method == "log_return":
            return reference_price * np.exp(prediction)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_target_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the target variable."""
        if "target" not in df.columns:
            df = self.transform(df)
        
        target = df["target"].dropna()
        
        return {
            "method": self.method,
            "count": int(len(target)),
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
            "median": float(target.median()),
            "skewness": float(target.skew()),
            "kurtosis": float(target.kurtosis()),
        }


def create_target(
    df: pd.DataFrame,
    method: Literal["none", "log", "return", "log_return"] = "none",
    **kwargs,
) -> Tuple[pd.DataFrame, TargetTransformer]:
    """
    Convenience function to create target variable.
    
    Args:
        df: DataFrame with 'close' column
        method: Transformation method
        **kwargs: Additional arguments for TargetTransformer
        
    Returns:
        Tuple of (transformed DataFrame, fitted transformer)
    """
    transformer = TargetTransformer(method=method, **kwargs)
    transformer.fit(df)
    df_transformed = transformer.transform(df)
    
    return df_transformed, transformer
