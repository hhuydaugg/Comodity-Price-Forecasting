"""
Data Loader for Commodity Price Data

Handles loading raw commodity data from various sources (CSV, Parquet, API)
and standardizes the schema to (date, close, commodity_id).
"""

from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import pandas as pd
import yaml
from loguru import logger


class CommodityLoader:
    """Load and standardize commodity price data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the loader with configuration.
        
        Args:
            config_path: Path to commodities.yaml config file.
                        If None, uses default config location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "commodities.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.commodities = {c["id"]: c for c in self.config.get("commodities", [])}
        self.schema = self.config.get("schema", {})
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def load_commodity(
        self,
        commodity_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data for a single commodity.
        
        Args:
            commodity_id: ID of the commodity to load
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: date, close, commodity_id
        """
        if commodity_id not in self.commodities:
            raise ValueError(f"Unknown commodity: {commodity_id}. "
                           f"Available: {list(self.commodities.keys())}")
        
        commodity_config = self.commodities[commodity_id]
        source = commodity_config.get("source", "file")
        
        if source == "file":
            df = self._load_from_file(commodity_config)
        else:
            raise ValueError(f"Unsupported source type: {source}")
        
        # Standardize schema
        df = self._standardize_schema(df, commodity_id)
        
        # Apply date filters
        df = self._filter_dates(df, start_date, end_date)
        
        logger.info(f"Loaded {len(df)} records for {commodity_id}")
        return df
    
    def _load_from_file(self, commodity_config: dict) -> pd.DataFrame:
        """Load data from a file (CSV or Parquet)."""
        file_path = Path(commodity_config["file_path"])
        
        # Handle relative paths
        if not file_path.is_absolute():
            file_path = self.config_path.parent.parent / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".csv":
            df = pd.read_csv(file_path, comment="#")
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return df
    
    def _standardize_schema(self, df: pd.DataFrame, commodity_id: str) -> pd.DataFrame:
        """Standardize DataFrame to expected schema."""
        date_col = self.schema.get("date_column", "date")
        close_col = self.schema.get("close_column", "close")
        date_format = self.schema.get("date_format", "%Y-%m-%d")
        
        # Find date column (case-insensitive)
        date_col_found = self._find_column(df, date_col)
        close_col_found = self._find_column(df, close_col)
        
        if date_col_found is None:
            raise ValueError(f"Date column '{date_col}' not found. Columns: {df.columns.tolist()}")
        if close_col_found is None:
            raise ValueError(f"Close column '{close_col}' not found. Columns: {df.columns.tolist()}")
        
        # Create standardized DataFrame
        result = pd.DataFrame({
            "date": pd.to_datetime(df[date_col_found], format=date_format, errors="coerce"),
            "close": pd.to_numeric(df[close_col_found], errors="coerce"),
            "commodity_id": commodity_id,
        })
        
        # Sort by date
        result = result.sort_values("date").reset_index(drop=True)
        
        return result
    
    def _find_column(self, df: pd.DataFrame, col_name: str) -> Optional[str]:
        """Find column name case-insensitively."""
        col_lower = col_name.lower()
        for col in df.columns:
            if col.lower() == col_lower:
                return col
        return None
    
    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df["date"] >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df["date"] <= end]
        
        return df.reset_index(drop=True)
    
    def load_all_commodities(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data for all configured commodities.
        
        Returns:
            DataFrame with data for all commodities concatenated
        """
        dfs = []
        
        for commodity_id in self.commodities:
            try:
                df = self.load_commodity(commodity_id, start_date, end_date)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {commodity_id}: {e}")
        
        if not dfs:
            return pd.DataFrame(columns=["date", "close", "commodity_id"])
        
        return pd.concat(dfs, ignore_index=True)
    
    def list_commodities(self) -> list[dict]:
        """List all available commodities with their metadata."""
        return list(self.commodities.values())
    
    def save_processed(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "parquet",
    ) -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('parquet' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(df)} records to {output_path}")


def load_commodity_data(
    commodity_id: str,
    config_path: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to load commodity data.
    
    Args:
        commodity_id: ID of the commodity
        config_path: Optional path to config file
        **kwargs: Additional arguments passed to load_commodity
        
    Returns:
        Standardized DataFrame
    """
    loader = CommodityLoader(config_path)
    return loader.load_commodity(commodity_id, **kwargs)
