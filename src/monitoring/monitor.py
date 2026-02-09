"""
Monitoring Module for Commodity Price Forecasting

Implements data drift detection, performance monitoring,
and retraining triggers.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DriftDetector:
    """Detect data distribution drift using statistical tests."""
    
    def __init__(
        self,
        reference_window: int = 60,  # Days for reference distribution
        test_window: int = 30,       # Days for test distribution
        psi_threshold: float = 0.2,  # PSI threshold for drift
        ks_threshold: float = 0.05,  # KS test p-value threshold
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_window: Number of days for reference distribution
            test_window: Number of days for test distribution
            psi_threshold: Population Stability Index threshold
            ks_threshold: KS test p-value threshold for significance
        """
        self.reference_window = reference_window
        self.test_window = test_window
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
    
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Slight change, monitor
        PSI > 0.2: Significant change, investigate
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins from reference data
        eps = 1e-8
        bins = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Calculate proportions
        ref_counts = np.histogram(reference, bins=bins)[0] + eps
        cur_counts = np.histogram(current, bins=bins)[0] + eps
        
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (statistic, p-value)
        """
        from scipy import stats
        
        statistic, pvalue = stats.ks_2samp(reference, current)
        return float(statistic), float(pvalue)
    
    def detect_drift(
        self,
        df: pd.DataFrame,
        column: str = "close",
    ) -> Dict:
        """
        Detect drift in the specified column.
        
        Args:
            df: DataFrame with 'date' and target column
            column: Column to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        df = df.sort_values("date")
        
        if len(df) < self.reference_window + self.test_window:
            return {"error": "Not enough data for drift detection"}
        
        # Split into reference and test
        reference = df[column].iloc[-(self.reference_window + self.test_window):-self.test_window].values
        current = df[column].iloc[-self.test_window:].values
        
        # Calculate metrics
        psi = self.calculate_psi(reference, current)
        ks_stat, ks_pvalue = self.ks_test(reference, current)
        
        # Determine if drift detected
        drift_detected = psi > self.psi_threshold or ks_pvalue < self.ks_threshold
        
        return {
            "drift_detected": drift_detected,
            "psi": psi,
            "psi_threshold": self.psi_threshold,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "ks_threshold": self.ks_threshold,
            "reference_period": f"{self.reference_window} days",
            "test_period": f"{self.test_window} days",
            "reference_mean": float(reference.mean()),
            "current_mean": float(current.mean()),
            "mean_change_pct": float((current.mean() - reference.mean()) / reference.mean() * 100),
        }


class PerformanceMonitor:
    """Monitor forecast performance over time."""
    
    def __init__(
        self,
        alert_threshold: float = 0.3,  # Alert if MASE > threshold
        degradation_threshold: float = 0.2,  # Alert if performance degrades by this %
        window_size: int = 30,  # Window for rolling performance
    ):
        """Initialize performance monitor."""
        self.alert_threshold = alert_threshold
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size
        self.history: List[Dict] = []
    
    def record_prediction(
        self,
        commodity_id: str,
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[str] = None,
        horizon: int = 1,
    ) -> None:
        """Record a prediction with optional actual value."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.history.append({
            "timestamp": timestamp,
            "commodity_id": commodity_id,
            "horizon": horizon,
            "prediction": prediction,
            "actual": actual,
            "error": abs(prediction - actual) if actual is not None else None,
        })
    
    def update_actuals(
        self,
        commodity_id: str,
        actuals: List[Tuple[str, float]],  # List of (timestamp, actual)
    ) -> int:
        """
        Update actual values for past predictions.
        
        Returns:
            Number of records updated
        """
        updated = 0
        actual_dict = {ts: val for ts, val in actuals}
        
        for record in self.history:
            if record["commodity_id"] == commodity_id and record["actual"] is None:
                if record["timestamp"] in actual_dict:
                    record["actual"] = actual_dict[record["timestamp"]]
                    record["error"] = abs(record["prediction"] - record["actual"])
                    updated += 1
        
        return updated
    
    def get_performance_summary(
        self,
        commodity_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get performance summary."""
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        if commodity_id:
            df = df[df["commodity_id"] == commodity_id]
        
        # Filter to records with actuals
        df = df[df["actual"].notna()]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Calculate metrics
        summary = df.groupby("commodity_id").agg({
            "error": ["mean", "std", "count"],
            "prediction": "mean",
            "actual": "mean",
        }).round(4)
        
        return summary
    
    def check_alerts(
        self,
        commodity_id: str,
    ) -> List[Dict]:
        """Check for performance alerts."""
        alerts = []
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return alerts
        
        df = df[df["commodity_id"] == commodity_id]
        df = df[df["actual"].notna()]
        
        if len(df) < self.window_size:
            return alerts
        
        # Recent performance
        recent = df.iloc[-self.window_size:]
        recent_mae = recent["error"].mean()
        recent_std = recent["error"].std()
        
        # Historical performance
        if len(df) >= self.window_size * 2:
            historical = df.iloc[-self.window_size * 2:-self.window_size]
            historical_mae = historical["error"].mean()
            
            # Check for degradation
            if historical_mae > 0:
                degradation = (recent_mae - historical_mae) / historical_mae
                if degradation > self.degradation_threshold:
                    alerts.append({
                        "type": "performance_degradation",
                        "commodity_id": commodity_id,
                        "severity": "warning",
                        "message": f"Performance degraded by {degradation:.1%}",
                        "details": {
                            "recent_mae": recent_mae,
                            "historical_mae": historical_mae,
                            "degradation_pct": degradation,
                        },
                    })
        
        return alerts


class RetrainTrigger:
    """Determine when to retrain models."""
    
    def __init__(
        self,
        schedule_days: int = 30,  # Retrain every N days by default
        performance_threshold: float = 0.25,  # Retrain if MASE exceeds this
        drift_trigger: bool = True,  # Retrain on data drift
    ):
        """Initialize retrain trigger."""
        self.schedule_days = schedule_days
        self.performance_threshold = performance_threshold
        self.drift_trigger = drift_trigger
        self.last_train_dates: Dict[str, datetime] = {}
    
    def record_training(self, commodity_id: str, date: Optional[datetime] = None) -> None:
        """Record when a model was trained."""
        self.last_train_dates[commodity_id] = date or datetime.now()
    
    def should_retrain(
        self,
        commodity_id: str,
        performance_metrics: Optional[Dict] = None,
        drift_detected: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if retraining is needed.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []
        
        # Check schedule
        last_train = self.last_train_dates.get(commodity_id)
        if last_train:
            days_since = (datetime.now() - last_train).days
            if days_since >= self.schedule_days:
                reasons.append(f"Scheduled retrain ({days_since} days since last)")
        else:
            reasons.append("No training record found")
        
        # Check performance
        if performance_metrics:
            mase = performance_metrics.get("mase", 0)
            if mase > self.performance_threshold:
                reasons.append(f"Performance below threshold (MASE={mase:.3f})")
        
        # Check drift
        if self.drift_trigger and drift_detected:
            reasons.append("Data drift detected")
        
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retrain needed"
        
        return should_retrain, reason
