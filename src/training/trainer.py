"""
Training Pipeline for Commodity Price Forecasting

Orchestrates data loading, preprocessing, feature engineering,
model training, and evaluation with MLflow tracking.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Local imports
from src.ingestion.loader import CommodityLoader
from src.ingestion.validator import DataValidator, validate_data
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.transformer import TargetTransformer
from src.preprocessing.splitter import TimeSeriesSplitter
from src.features.generator import FeatureGenerator
from src.models.baseline import get_baseline_models
from src.models.ml import XGBoostForecaster, LightGBMForecaster
from src.evaluation.backtest import TimeSeriesBacktest, compare_models
from src.evaluation.metrics import calculate_metrics


class Trainer:
    """
    Main training pipeline orchestrator.
    
    Handles the complete workflow from data loading to model evaluation.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
        use_mlflow: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to commodities.yaml
            model_config_path: Path to model_config.yaml
            use_mlflow: Whether to use MLflow for tracking
        """
        self.base_path = Path(__file__).parent.parent.parent
        
        if config_path is None:
            config_path = self.base_path / "configs" / "commodities.yaml"
        if model_config_path is None:
            model_config_path = self.base_path / "configs" / "model_config.yaml"
        
        self.config = self._load_yaml(config_path)
        self.model_config = self._load_yaml(model_config_path)
        
        self.use_mlflow = use_mlflow
        
        # Initialize components
        self.loader = CommodityLoader(config_path)
        self.validator = DataValidator(config_path)
        self.cleaner = DataCleaner(
            frequency="auto",  # Auto-detect calendar vs business days
            fill_method="ffill",
            max_gap_days=10,
        )
        self.feature_generator = FeatureGenerator(
            lag_days=self.model_config.get("features", {}).get("lag_days", [1, 2, 7, 14, 30]),
            rolling_windows=self.model_config.get("features", {}).get("rolling_windows", [7, 14, 30]),
        )
        
        # Train/test split config
        split_config = self.model_config.get("train_test_split", {})
        self.splitter = TimeSeriesSplitter(
            train_ratio=split_config.get("train_ratio", 0.8),
            min_test_days=split_config.get("min_test_days", 30),
            gap_days=split_config.get("gap_days", 0),
        )
        
        # Target transformation
        target_type = self.model_config.get("forecast", {}).get("target", "close")
        self.transformer = TargetTransformer(
            method="none" if target_type == "close" else target_type
        )
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _setup_mlflow(self, experiment_name: str) -> None:
        """Setup MLflow tracking."""
        if not self.use_mlflow:
            return
        
        try:
            import mlflow
            
            tracking_uri = self.model_config.get("mlflow", {}).get("tracking_uri", "mlruns")
            mlflow.set_tracking_uri(str(self.base_path / tracking_uri))
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
            
        except ImportError:
            logger.warning("MLflow not installed, tracking disabled")
            self.use_mlflow = False
    
    def prepare_data(
        self,
        commodity_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and prepare data for training.
        
        Args:
            commodity_id: ID of commodity to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Prepared DataFrame with features
        """
        # 1. Load raw data
        logger.info(f"Loading data for {commodity_id}")
        df = self.loader.load_commodity(commodity_id, start_date, end_date)
        
        # 2. Validate
        report = self.validator.validate(df, commodity_id)
        if not report.passed:
            logger.warning(f"Validation warnings:\n{report.summary()}")
        
        # 3. Clean
        logger.info("Cleaning data")
        df = self.cleaner.clean(df)
        
        # 4. Transform target
        df = self.transformer.transform(df, commodity_id)
        
        # 5. Generate features
        logger.info("Generating features")
        df = self.feature_generator.generate(df)
        
        return df
    
    def train_commodity(
        self,
        commodity_id: str,
        model_type: Literal["baseline", "statistical", "ml", "all"] = "all",
        backtest: bool = True,
        save_model: bool = True,
    ) -> Dict:
        """
        Train models for a single commodity.
        
        Args:
            commodity_id: ID of commodity
            model_type: Which models to train
            backtest: Whether to run backtest evaluation
            save_model: Whether to save best model
            
        Returns:
            Dictionary with training results
        """
        self._setup_mlflow(
            self.model_config.get("mlflow", {}).get("experiment_name", "commodity_forecast")
        )
        
        # Prepare data
        df = self.prepare_data(commodity_id)
        
        if len(df) < 60:
            raise ValueError(f"Not enough data for {commodity_id}: {len(df)} rows (need >= 60)")
        
        # Get feature columns (excluding metadata)
        feature_cols = [c for c in self.feature_generator.get_feature_names()
                       if c in df.columns]
        
        # Filter out calendar features for ML and keep only lag/rolling/volatility
        ml_feature_cols = [c for c in feature_cols 
                          if not c.startswith("is_dow_") and c not in ["date"]]
        
        results = {
            "commodity_id": commodity_id,
            "data_rows": len(df),
            "n_features": len(ml_feature_cols),
            "models": {},
        }
        
        # Train models
        if model_type in ["baseline", "all"]:
            results["models"]["baseline"] = self._train_baseline(df, backtest)
        
        if model_type in ["ml", "all"]:
            results["models"]["ml"] = self._train_ml(
                df, ml_feature_cols, commodity_id, backtest, save_model
            )
        
        return results
    
    def _train_baseline(self, df: pd.DataFrame, backtest: bool) -> Dict:
        """Train and evaluate baseline models."""
        from src.models.baseline import NaiveModel, SeasonalNaiveModel
        
        baselines = [
            NaiveModel(),
            SeasonalNaiveModel(season_length=7),
        ]
        
        if backtest:
            bt_config = self.model_config.get("backtesting", {})
            comparison = compare_models(
                df=df.dropna(subset=["close"]),
                models=baselines,
                target_col="close",
                initial_train_size=bt_config.get("initial_train_days", 504),
                test_size=bt_config.get("test_days", 30),
                step_size=bt_config.get("step_days", 30),
            )
            return comparison.to_dict("records")
        
        return {"models": [m.name for m in baselines]}
    
    def _train_ml(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        commodity_id: str,
        backtest: bool,
        save_model: bool,
    ) -> Dict:
        """Train and evaluate ML models."""
        # Prepare data
        df_clean = df.dropna(subset=feature_cols + ["close"]).copy()
        
        if len(df_clean) < 100:
            logger.warning(f"Not enough clean data for ML: {len(df_clean)} rows")
            return {"error": "Insufficient data"}
        
        # Use TimeSeriesSplitter for proper chronological split
        split_result = self.splitter.split(df_clean)
        train_df = split_result.train_df
        test_df = split_result.test_df
        logger.info(f"Train/test split:\n{split_result.summary()}")
        
        X_train = train_df[feature_cols]
        y_train = train_df["close"]
        X_test = test_df[feature_cols]
        y_test = test_df["close"]
        
        results = {}
        best_model = None
        best_metric = float("inf")
        primary_metric = self.model_config.get("evaluation", {}).get("primary_metric", "mae")
        
        # XGBoost
        try:
            xgb_config = self.model_config.get("models", {}).get("xgboost", {})
            if xgb_config.get("enabled", True):
                xgb_model = XGBoostForecaster(**xgb_config.get("params", {}))
                
                if self.use_mlflow:
                    import mlflow
                    with mlflow.start_run(run_name=f"xgboost_{commodity_id}"):
                        xgb_model.fit(y_train, X_train)
                        preds = xgb_model.predict(X=X_test)
                        metrics = calculate_metrics(y_test.values, preds, y_train.values)
                        
                        mlflow.log_params(xgb_model.get_params())
                        mlflow.log_metrics(metrics)
                        
                        results["xgboost"] = metrics
                        
                        if metrics.get(primary_metric, float("inf")) < best_metric:
                            best_metric = metrics[primary_metric]
                            best_model = xgb_model
                else:
                    xgb_model.fit(y_train, X_train)
                    preds = xgb_model.predict(X=X_test)
                    metrics = calculate_metrics(y_test.values, preds, y_train.values)
                    results["xgboost"] = metrics
                    
                    if metrics.get(primary_metric, float("inf")) < best_metric:
                        best_metric = metrics[primary_metric]
                        best_model = xgb_model
                        
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results["xgboost"] = {"error": str(e)}
        
        # LightGBM
        try:
            lgb_config = self.model_config.get("models", {}).get("lightgbm", {})
            if lgb_config.get("enabled", True):
                lgb_model = LightGBMForecaster(**lgb_config.get("params", {}))
                
                if self.use_mlflow:
                    import mlflow
                    with mlflow.start_run(run_name=f"lightgbm_{commodity_id}"):
                        lgb_model.fit(y_train, X_train)
                        preds = lgb_model.predict(X=X_test)
                        metrics = calculate_metrics(y_test.values, preds, y_train.values)
                        
                        mlflow.log_params(lgb_model.get_params())
                        mlflow.log_metrics(metrics)
                        
                        results["lightgbm"] = metrics
                        
                        if metrics.get(primary_metric, float("inf")) < best_metric:
                            best_metric = metrics[primary_metric]
                            best_model = lgb_model
                else:
                    lgb_model.fit(y_train, X_train)
                    preds = lgb_model.predict(X=X_test)
                    metrics = calculate_metrics(y_test.values, preds, y_train.values)
                    results["lightgbm"] = metrics
                    
                    if metrics.get(primary_metric, float("inf")) < best_metric:
                        best_metric = metrics[primary_metric]
                        best_model = lgb_model
                        
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            results["lightgbm"] = {"error": str(e)}
        
        # Save best model
        if save_model and best_model is not None:
            model_dir = self.base_path / "models" / commodity_id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{best_model.name}_best.model"
            best_model.save(model_path)
            results["best_model"] = {
                "name": best_model.name,
                "path": str(model_path),
                f"{primary_metric}": best_metric,
            }
        
        return results
    
    def train_all(
        self,
        commodity_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        Train models for all commodities.
        
        Args:
            commodity_ids: List of commodities to train (None for all)
            **kwargs: Arguments passed to train_commodity
            
        Returns:
            List of results for each commodity
        """
        if commodity_ids is None:
            commodity_ids = list(self.loader.commodities.keys())
        
        all_results = []
        for commodity_id in commodity_ids:
            try:
                logger.info(f"Training {commodity_id}")
                result = self.train_commodity(commodity_id, **kwargs)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to train {commodity_id}: {e}")
                all_results.append({
                    "commodity_id": commodity_id,
                    "error": str(e),
                })
        
        return all_results


def main():
    """CLI entry point for training."""
    import click
    
    @click.command()
    @click.option("--commodity", "-c", default="all", help="Commodity ID or 'all'")
    @click.option("--model-type", "-m", default="all", 
                  type=click.Choice(["baseline", "statistical", "ml", "all"]))
    @click.option("--no-mlflow", is_flag=True, help="Disable MLflow tracking")
    @click.option("--no-save", is_flag=True, help="Don't save models")
    def train(commodity, model_type, no_mlflow, no_save):
        """Train commodity price forecasting models."""
        trainer = Trainer(use_mlflow=not no_mlflow)
        
        if commodity == "all":
            results = trainer.train_all(
                model_type=model_type,
                save_model=not no_save,
            )
        else:
            results = [trainer.train_commodity(
                commodity_id=commodity,
                model_type=model_type,
                save_model=not no_save,
            )]
        
        # Print summary
        for r in results:
            logger.info(f"\n{r.get('commodity_id', 'Unknown')}:")
            if "error" in r:
                logger.error(f"  Error: {r['error']}")
            else:
                logger.info(f"  Data rows: {r.get('data_rows', 'N/A')}")
                logger.info(f"  Features: {r.get('n_features', 'N/A')}")
                if "models" in r and "ml" in r["models"]:
                    if "best_model" in r["models"]["ml"]:
                        best = r["models"]["ml"]["best_model"]
                        logger.info(f"  Best model: {best['name']}")
    
    train()


if __name__ == "__main__":
    main()
