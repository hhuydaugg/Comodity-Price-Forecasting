"""
Visualization utilities for commodity price predictions.

Provides functions to plot predictions vs ground truth, forecast horizons,
and model performance metrics.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import seaborn as sns

from src.ingestion.loader import CommodityLoader


class PredictionVisualizer:
    """Visualize commodity price predictions and ground truth."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        # Set color palette
        self.colors = {
            "actual": "#2E86AB",      # Blue
            "prediction": "#A23B72",  # Purple
            "confidence": "#F18F01",  # Orange
            "horizon_1": "#06A77D",   # Green
            "horizon_7": "#F18F01",   # Orange
            "horizon_30": "#C73E1D",  # Red
        }
    
    def plot_prediction_vs_actual(
        self,
        commodity_id: str,
        predictions_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame] = None,
        lookback_days: int = 90,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot predictions against actual prices.
        
        Args:
            commodity_id: ID of the commodity
            predictions_df: DataFrame with predictions (from predictor)
            actual_df: DataFrame with actual prices (if None, loads from data)
            lookback_days: Number of historical days to show
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Load actual data if not provided
        if actual_df is None:
            loader = CommodityLoader()
            actual_df = loader.load_commodity(commodity_id)
        
        # Filter predictions for this commodity
        preds = predictions_df[predictions_df["commodity_id"] == commodity_id].copy()
        
        if len(preds) == 0:
            raise ValueError(f"No predictions found for {commodity_id}")
        
        # Get the as_of_date (reference date for predictions)
        as_of_date = pd.to_datetime(preds["as_of_date"].iloc[0])
        
        # Filter historical data
        historical = actual_df[actual_df["date"] <= as_of_date].copy()
        if lookback_days > 0:
            cutoff_date = as_of_date - pd.Timedelta(days=lookback_days)
            historical = historical[historical["date"] >= cutoff_date]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical prices
        ax.plot(
            historical["date"],
            historical["close"],
            label="Historical Prices",
            color=self.colors["actual"],
            linewidth=2,
            marker="o",
            markersize=3,
        )
        
        # Plot predictions for each horizon
        horizon_colors = {
            1: self.colors["horizon_1"],
            7: self.colors["horizon_7"],
            30: self.colors["horizon_30"],
        }
        
        for _, pred_row in preds.iterrows():
            horizon = pred_row["horizon"]
            target_date = pd.to_datetime(pred_row["target_date"])
            prediction = pred_row["prediction"]
            
            color = horizon_colors.get(horizon, self.colors["prediction"])
            
            # Draw line from as_of_date to target_date
            ax.plot(
                [as_of_date, target_date],
                [pred_row["last_close"], prediction],
                label=f"{horizon}-day forecast" if horizon in [1, 7, 30] else None,
                color=color,
                linewidth=2,
                linestyle="--",
                marker="o",
                markersize=6,
            )
            
            # Add prediction value annotation
            ax.annotate(
                f"${prediction:.2f}",
                xy=(target_date, prediction),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8),
            )
        
        # Mark the as_of_date with a vertical line
        ax.axvline(
            as_of_date,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Prediction Date",
            alpha=0.7,
        )
        
        # Formatting
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price ($)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{commodity_id.replace('_', ' ').title()} - Price Predictions",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_multi_horizon_comparison(
        self,
        commodity_id: str,
        predictions_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Plot comparison of different forecast horizons.
        
        Args:
            commodity_id: ID of the commodity
            predictions_df: DataFrame with predictions
            actual_df: DataFrame with actual prices
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Load actual data if not provided
        if actual_df is None:
            loader = CommodityLoader()
            actual_df = loader.load_commodity(commodity_id)
        
        # Filter predictions
        preds = predictions_df[predictions_df["commodity_id"] == commodity_id].copy()
        preds["target_date"] = pd.to_datetime(preds["target_date"])
        
        # Get unique horizons
        horizons = sorted(preds["horizon"].unique())
        
        # Create subplots
        n_horizons = len(horizons)
        fig, axes = plt.subplots(n_horizons, 1, figsize=figsize, sharex=True)
        
        if n_horizons == 1:
            axes = [axes]
        
        as_of_date = pd.to_datetime(preds["as_of_date"].iloc[0])
        
        for idx, horizon in enumerate(horizons):
            ax = axes[idx]
            
            # Get predictions for this horizon
            horizon_preds = preds[preds["horizon"] == horizon]
            
            # Plot historical data
            historical = actual_df[actual_df["date"] <= as_of_date]
            lookback = historical.tail(60)  # Last 60 days
            
            ax.plot(
                lookback["date"],
                lookback["close"],
                label="Historical",
                color=self.colors["actual"],
                linewidth=2,
            )
            
            # Plot prediction
            for _, pred_row in horizon_preds.iterrows():
                target_date = pred_row["target_date"]
                prediction = pred_row["prediction"]
                
                ax.plot(
                    [as_of_date, target_date],
                    [pred_row["last_close"], prediction],
                    label=f"Forecast",
                    color=self.colors["prediction"],
                    linewidth=2,
                    linestyle="--",
                    marker="o",
                )
                
                # Check if we have ground truth for this date
                actual_at_target = actual_df[actual_df["date"] == target_date]
                if len(actual_at_target) > 0:
                    actual_price = actual_at_target["close"].iloc[0]
                    ax.scatter(
                        target_date,
                        actual_price,
                        color=self.colors["actual"],
                        s=100,
                        zorder=5,
                        label="Actual",
                        edgecolors="black",
                        linewidths=1.5,
                    )
                    
                    # Add error annotation
                    error = prediction - actual_price
                    error_pct = (error / actual_price) * 100
                    ax.annotate(
                        f"Error: {error_pct:+.2f}%",
                        xy=(target_date, actual_price),
                        xytext=(10, -10),
                        textcoords="offset points",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    )
            
            ax.axvline(as_of_date, color="gray", linestyle=":", alpha=0.7)
            ax.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
            ax.set_title(f"{horizon}-Day Horizon", fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Date", fontsize=12, fontweight="bold")
        fig.suptitle(
            f"{commodity_id.replace('_', ' ').title()} - Multi-Horizon Forecasts",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_forecast_accuracy(
        self,
        predictions_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot forecast accuracy metrics across horizons.
        
        Args:
            predictions_df: DataFrame with predictions
            actual_df: DataFrame with actual prices
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate errors
        errors = []
        
        for _, pred_row in predictions_df.iterrows():
            target_date = pd.to_datetime(pred_row["target_date"])
            actual_at_target = actual_df[actual_df["date"] == target_date]
            
            if len(actual_at_target) > 0:
                actual_price = actual_at_target["close"].iloc[0]
                prediction = pred_row["prediction"]
                
                error = prediction - actual_price
                abs_error = abs(error)
                pct_error = (error / actual_price) * 100
                abs_pct_error = abs(pct_error)
                
                errors.append({
                    "commodity_id": pred_row["commodity_id"],
                    "horizon": pred_row["horizon"],
                    "error": error,
                    "abs_error": abs_error,
                    "pct_error": pct_error,
                    "abs_pct_error": abs_pct_error,
                })
        
        if not errors:
            raise ValueError("No matching actual data found for predictions")
        
        errors_df = pd.DataFrame(errors)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Mean Absolute Percentage Error by Horizon
        horizon_stats = errors_df.groupby("horizon").agg({
            "abs_pct_error": ["mean", "std"],
        }).reset_index()
        horizon_stats.columns = ["horizon", "mape", "std"]
        
        ax1.bar(
            horizon_stats["horizon"],
            horizon_stats["mape"],
            yerr=horizon_stats["std"],
            color=self.colors["prediction"],
            alpha=0.7,
            capsize=5,
        )
        ax1.set_xlabel("Forecast Horizon (days)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("MAPE (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Mean Absolute Percentage Error", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        
        # Plot 2: Error distribution by horizon
        horizons = sorted(errors_df["horizon"].unique())
        positions = range(len(horizons))
        
        box_data = [errors_df[errors_df["horizon"] == h]["pct_error"].values for h in horizons]
        
        bp = ax2.boxplot(
            box_data,
            positions=positions,
            labels=horizons,
            patch_artist=True,
            notch=True,
        )
        
        for patch in bp["boxes"]:
            patch.set_facecolor(self.colors["prediction"])
            patch.set_alpha(0.7)
        
        ax2.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax2.set_xlabel("Forecast Horizon (days)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Percentage Error (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Error Distribution", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        
        return fig


def plot_predictions_quick(
    commodity_id: str,
    predictions_path: str,
    lookback_days: int = 90,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Quick function to plot predictions from a saved file.
    
    Args:
        commodity_id: ID of the commodity
        predictions_path: Path to predictions CSV/Parquet
        lookback_days: Number of historical days to show
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Load predictions
    predictions_path = Path(predictions_path)
    if predictions_path.suffix == ".csv":
        predictions_df = pd.read_csv(predictions_path)
    else:
        predictions_df = pd.read_parquet(predictions_path)
    
    # Create visualizer and plot
    viz = PredictionVisualizer()
    fig = viz.plot_prediction_vs_actual(
        commodity_id=commodity_id,
        predictions_df=predictions_df,
        lookback_days=lookback_days,
        save_path=save_path,
    )
    
    return fig


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plot_predictions.py <commodity_id> <predictions_file> [output_file]")
        sys.exit(1)
    
    commodity_id = sys.argv[1]
    predictions_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    fig = plot_predictions_quick(commodity_id, predictions_file, save_path=output_file)
    
    if output_file is None:
        plt.show()
