"""
Batch Predictor for Commodity Price Forecasting

Loads trained models and generates predictions for production use.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.ingestion.loader import CommodityLoader
from src.preprocessing.cleaner import DataCleaner
from src.features.generator import FeatureGenerator


class BatchPredictor:
    """
    Generate batch predictions for all commodities.
    
    Loads trained models and produces forecasts for production deployment.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            config_path: Path to commodities.yaml
            model_dir: Directory containing trained models
        """
        self.base_path = Path(__file__).parent.parent.parent
        
        if config_path is None:
            config_path = self.base_path / "configs" / "commodities.yaml"
        if model_dir is None:
            model_dir = self.base_path / "models"
        
        self.config_path = Path(config_path)
        self.model_dir = Path(model_dir)
        
        self.config = self._load_yaml(config_path)
        self.model_config = self._load_yaml(self.base_path / "configs" / "model_config.yaml")
        
        self.loader = CommodityLoader(config_path)
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator(
            lag_days=self.model_config.get("features", {}).get("lag_days", [1, 2, 7, 14, 30]),
            rolling_windows=self.model_config.get("features", {}).get("rolling_windows", [7, 14, 30]),
        )
        
        self._models: Dict = {}
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file."""
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def load_model(self, commodity_id: str, model_type: str = "xgboost") -> bool:
        """
        Load a trained model for a commodity.
        
        Args:
            commodity_id: ID of the commodity
            model_type: Type of model (xgboost, lightgbm)
            
        Returns:
            True if model loaded successfully
        """
        model_path = self.model_dir / commodity_id / f"{model_type}_best.model"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False
        
        try:
            if model_type == "xgboost":
                from src.models.ml import XGBoostForecaster
                model = XGBoostForecaster()
                model.load(model_path)
            elif model_type == "lightgbm":
                from src.models.ml import LightGBMForecaster
                model = LightGBMForecaster()
                model.load(model_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            self._models[commodity_id] = model
            logger.info(f"Loaded {model_type} model for {commodity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_commodity(
        self,
        commodity_id: str,
        horizons: List[int] = None,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for a single commodity.
        
        Args:
            commodity_id: ID of the commodity
            horizons: List of forecast horizons (days ahead)
            as_of_date: Reference date for prediction (default: latest data)
            
        Returns:
            DataFrame with predictions
        """
        if horizons is None:
            horizons = self.model_config.get("forecast", {}).get("horizons", [1, 7, 30])
        
        # Load model if not already loaded
        if commodity_id not in self._models:
            if not self.load_model(commodity_id):
                raise ValueError(f"No model available for {commodity_id}")
        
        model = self._models[commodity_id]
        
        # Load and prepare recent data
        df = self.loader.load_commodity(commodity_id)
        df = self.cleaner.clean(df)
        df = self.feature_generator.generate(df)
        
        # Get feature columns
        feature_cols = [c for c in self.feature_generator.get_feature_names()
                       if c in df.columns and not c.startswith("is_dow_")]
        
        # Get latest data point
        df_clean = df.dropna(subset=feature_cols)
        if len(df_clean) == 0:
            raise ValueError("No valid data for prediction")
        
        latest = df_clean.iloc[-1]
        latest_date = latest["date"]
        latest_price = latest["close"]
        
        # Generate predictions for each horizon
        results = []
        X = df_clean[feature_cols].iloc[[-1]]  # Single row DataFrame
        
        for h in horizons:
            try:
                # For multi-step ahead, we need to iterate (simplified version)
                pred = model.predict(X=X)[0]
                
                results.append({
                    "commodity_id": commodity_id,
                    "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "as_of_date": str(latest_date),
                    "horizon": h,
                    "target_date": (pd.Timestamp(latest_date) + timedelta(days=h)).strftime("%Y-%m-%d"),
                    "prediction": float(pred),
                    "last_close": float(latest_price),
                    "predicted_change_pct": float((pred - latest_price) / latest_price * 100),
                })
            except Exception as e:
                logger.error(f"Prediction failed for horizon {h}: {e}")
        
        return pd.DataFrame(results)
    
    def predict_all(
        self,
        commodity_ids: Optional[List[str]] = None,
        horizons: List[int] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for all commodities.
        
        Args:
            commodity_ids: List of commodities (None for all)
            horizons: Forecast horizons
            
        Returns:
            DataFrame with all predictions
        """
        if commodity_ids is None:
            commodity_ids = list(self.loader.commodities.keys())
        
        all_predictions = []
        
        for commodity_id in commodity_ids:
            try:
                preds = self.predict_commodity(commodity_id, horizons)
                all_predictions.append(preds)
            except Exception as e:
                logger.error(f"Failed to predict {commodity_id}: {e}")
        
        if not all_predictions:
            return pd.DataFrame()
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: Optional[str] = None,
        format: Literal["parquet", "csv"] = "parquet",
    ) -> str:
        """
        Save predictions to file.
        
        Args:
            predictions: Predictions DataFrame
            output_path: Output file path (auto-generated if None)
            format: Output format
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_dir = self.base_path / "data" / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"predictions_{timestamp}.{format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            predictions.to_parquet(output_path, index=False)
        else:
            predictions.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        return str(output_path)


def main():
    """CLI entry point for batch prediction."""
    import click
    
    @click.command()
    @click.option("--commodity", "-c", default="all", help="Commodity ID or 'all'")
    @click.option("--horizon", "-h", multiple=True, type=int, default=[1, 7, 30])
    @click.option("--output", "-o", default=None, help="Output file path")
    @click.option("--format", "-f", default="parquet", type=click.Choice(["parquet", "csv"]))
    def predict(commodity, horizon, output, format):
        """Generate batch predictions for commodity prices."""
        predictor = BatchPredictor()
        
        horizons = list(horizon)
        
        if commodity == "all":
            predictions = predictor.predict_all(horizons=horizons)
        else:
            predictions = predictor.predict_commodity(commodity, horizons=horizons)
        
        if len(predictions) > 0:
            path = predictor.save_predictions(predictions, output, format)
            click.echo(f"Predictions saved to: {path}")
            click.echo(predictions.to_string(index=False))
        else:
            click.echo("No predictions generated")
    
    predict()


if __name__ == "__main__":
    main()
