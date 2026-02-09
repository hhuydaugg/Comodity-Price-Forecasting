"""
Data Validator for Commodity Price Data

Implements quality gates and validation checks for raw/processed data.
Validates: date uniqueness, close values, missing data, outliers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    
    commodity_id: str
    timestamp: datetime
    results: list[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all error-level validations passed."""
        return all(
            r.passed for r in self.results 
            if r.severity == "error"
        )
    
    @property
    def error_count(self) -> int:
        """Count of failed error-level validations."""
        return sum(1 for r in self.results if not r.passed and r.severity == "error")
    
    @property
    def warning_count(self) -> int:
        """Count of failed warning-level validations."""
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "commodity_id": self.commodity_id,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Validation Report for {self.commodity_id}",
            f"Status: {status}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
            "-" * 50,
        ]
        
        for r in self.results:
            icon = "✅" if r.passed else ("❌" if r.severity == "error" else "⚠️")
            lines.append(f"{icon} {r.check_name}: {r.message}")
        
        return "\n".join(lines)


class DataValidator:
    """Validate commodity price data quality."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize validator with configuration.
        
        Args:
            config_path: Path to commodities.yaml with validation thresholds
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "commodities.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.validation_config = self.config.get("validation", {})
        
        # Default thresholds
        self.max_missing_pct = self.validation_config.get("max_missing_days_pct", 0.05)
        self.outlier_threshold = self.validation_config.get("outlier_std_threshold", 4.0)
        self.min_history_days = self.validation_config.get("min_history_days", 252)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def validate(
        self,
        df: pd.DataFrame,
        commodity_id: str = "unknown",
        fail_fast: bool = False,
    ) -> ValidationReport:
        """
        Run all validation checks on the data.
        
        Args:
            df: DataFrame with columns (date, close, commodity_id)
            commodity_id: ID of the commodity for reporting
            fail_fast: If True, stop on first error
            
        Returns:
            ValidationReport with all check results
        """
        report = ValidationReport(
            commodity_id=commodity_id,
            timestamp=datetime.now(),
        )
        
        checks = [
            self._check_required_columns,
            self._check_date_parseable,
            self._check_date_unique,
            self._check_date_chronological,
            self._check_close_numeric,
            self._check_close_positive,
            self._check_close_not_null,
            self._check_missing_dates,
            self._check_outliers,
            self._check_min_history,
        ]
        
        for check in checks:
            try:
                result = check(df)
                report.results.append(result)
                
                if fail_fast and not result.passed and result.severity == "error":
                    logger.error(f"Validation failed: {result.message}")
                    break
                    
            except Exception as e:
                report.results.append(ValidationResult(
                    check_name=check.__name__,
                    passed=False,
                    message=f"Check failed with exception: {e}",
                    severity="error",
                ))
        
        return report
    
    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Check that required columns exist."""
        required = ["date", "close"]
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            return ValidationResult(
                check_name="required_columns",
                passed=False,
                message=f"Missing required columns: {missing}",
                details={"missing": missing, "available": df.columns.tolist()},
            )
        
        return ValidationResult(
            check_name="required_columns",
            passed=True,
            message="All required columns present",
        )
    
    def _check_date_parseable(self, df: pd.DataFrame) -> ValidationResult:
        """Check that all dates can be parsed."""
        if "date" not in df.columns:
            return ValidationResult(
                check_name="date_parseable",
                passed=False,
                message="Date column not found",
            )
        
        # Try to convert to datetime
        dates = pd.to_datetime(df["date"], errors="coerce")
        null_count = dates.isna().sum()
        
        if null_count > 0:
            return ValidationResult(
                check_name="date_parseable",
                passed=False,
                message=f"{null_count} dates could not be parsed",
                details={"unparseable_count": int(null_count)},
            )
        
        return ValidationResult(
            check_name="date_parseable",
            passed=True,
            message="All dates are parseable",
        )
    
    def _check_date_unique(self, df: pd.DataFrame) -> ValidationResult:
        """Check that dates are unique."""
        if "date" not in df.columns:
            return ValidationResult(
                check_name="date_unique",
                passed=False,
                message="Date column not found",
            )
        
        duplicates = df["date"].duplicated().sum()
        
        if duplicates > 0:
            dup_dates = df[df["date"].duplicated(keep=False)]["date"].unique()
            return ValidationResult(
                check_name="date_unique",
                passed=False,
                message=f"{duplicates} duplicate dates found",
                details={"duplicate_dates": [str(d) for d in dup_dates[:10]]},
            )
        
        return ValidationResult(
            check_name="date_unique",
            passed=True,
            message="All dates are unique",
        )
    
    def _check_date_chronological(self, df: pd.DataFrame) -> ValidationResult:
        """Check that dates are in chronological order."""
        if "date" not in df.columns or len(df) < 2:
            return ValidationResult(
                check_name="date_chronological",
                passed=True,
                message="Not enough data to check order",
            )
        
        dates = pd.to_datetime(df["date"])
        is_sorted = dates.is_monotonic_increasing
        
        if not is_sorted:
            return ValidationResult(
                check_name="date_chronological",
                passed=False,
                message="Dates are not in chronological order",
                severity="warning",  # Can be fixed by sorting
            )
        
        return ValidationResult(
            check_name="date_chronological",
            passed=True,
            message="Dates are in chronological order",
        )
    
    def _check_close_numeric(self, df: pd.DataFrame) -> ValidationResult:
        """Check that close values are numeric."""
        if "close" not in df.columns:
            return ValidationResult(
                check_name="close_numeric",
                passed=False,
                message="Close column not found",
            )
        
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df["close"]):
            return ValidationResult(
                check_name="close_numeric",
                passed=True,
                message="Close values are numeric",
            )
        
        # Try to convert
        converted = pd.to_numeric(df["close"], errors="coerce")
        non_numeric = converted.isna().sum() - df["close"].isna().sum()
        
        if non_numeric > 0:
            return ValidationResult(
                check_name="close_numeric",
                passed=False,
                message=f"{non_numeric} close values are not numeric",
                details={"non_numeric_count": int(non_numeric)},
            )
        
        return ValidationResult(
            check_name="close_numeric",
            passed=True,
            message="Close values are numeric",
        )
    
    def _check_close_positive(self, df: pd.DataFrame) -> ValidationResult:
        """Check that close values are positive."""
        if "close" not in df.columns:
            return ValidationResult(
                check_name="close_positive",
                passed=False,
                message="Close column not found",
            )
        
        close = pd.to_numeric(df["close"], errors="coerce")
        negative_count = (close < 0).sum()
        
        if negative_count > 0:
            return ValidationResult(
                check_name="close_positive",
                passed=False,
                message=f"{negative_count} negative close values found",
                details={"negative_count": int(negative_count)},
            )
        
        return ValidationResult(
            check_name="close_positive",
            passed=True,
            message="All close values are non-negative",
        )
    
    def _check_close_not_null(self, df: pd.DataFrame) -> ValidationResult:
        """Check that close values are not null."""
        if "close" not in df.columns:
            return ValidationResult(
                check_name="close_not_null",
                passed=False,
                message="Close column not found",
            )
        
        null_count = df["close"].isna().sum()
        null_pct = null_count / len(df) if len(df) > 0 else 0
        
        if null_count > 0:
            return ValidationResult(
                check_name="close_not_null",
                passed=False,
                message=f"{null_count} null close values ({null_pct:.1%})",
                details={"null_count": int(null_count), "null_pct": float(null_pct)},
            )
        
        return ValidationResult(
            check_name="close_not_null",
            passed=True,
            message="No null close values",
        )
    
    def _check_missing_dates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for missing dates in the series."""
        if "date" not in df.columns or len(df) < 2:
            return ValidationResult(
                check_name="missing_dates",
                passed=True,
                message="Not enough data to check missing dates",
                severity="warning",
            )
        
        dates = pd.to_datetime(df["date"]).sort_values()
        
        # Create expected date range (business days)
        expected = pd.bdate_range(start=dates.min(), end=dates.max())
        actual = set(dates.dt.date)
        expected_set = set(expected.date)
        
        missing = expected_set - actual
        missing_pct = len(missing) / len(expected_set) if expected_set else 0
        
        if missing_pct > self.max_missing_pct:
            return ValidationResult(
                check_name="missing_dates",
                passed=False,
                message=f"{len(missing)} missing dates ({missing_pct:.1%} > {self.max_missing_pct:.1%})",
                details={
                    "missing_count": len(missing),
                    "missing_pct": float(missing_pct),
                    "threshold": self.max_missing_pct,
                },
                severity="warning",
            )
        
        return ValidationResult(
            check_name="missing_dates",
            passed=True,
            message=f"{len(missing)} missing dates ({missing_pct:.1%})",
            details={"missing_count": len(missing), "missing_pct": float(missing_pct)},
        )
    
    def _check_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Check for outliers using rolling z-score."""
        if "close" not in df.columns or len(df) < 30:
            return ValidationResult(
                check_name="outliers",
                passed=True,
                message="Not enough data to detect outliers",
                severity="info",
            )
        
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        
        # Calculate rolling z-score
        rolling_mean = close.rolling(window=30, min_periods=10).mean()
        rolling_std = close.rolling(window=30, min_periods=10).std()
        
        z_scores = np.abs((close - rolling_mean) / rolling_std)
        outliers = z_scores > self.outlier_threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_dates = df.loc[outliers[outliers].index, "date"].tolist()
            return ValidationResult(
                check_name="outliers",
                passed=True,  # Outliers are warnings, not errors
                message=f"{outlier_count} potential outliers detected (z > {self.outlier_threshold})",
                details={
                    "outlier_count": int(outlier_count),
                    "outlier_dates": [str(d) for d in outlier_dates[:5]],
                },
                severity="warning",
            )
        
        return ValidationResult(
            check_name="outliers",
            passed=True,
            message="No significant outliers detected",
        )
    
    def _check_min_history(self, df: pd.DataFrame) -> ValidationResult:
        """Check that there's enough historical data."""
        if "date" not in df.columns:
            return ValidationResult(
                check_name="min_history",
                passed=False,
                message="Date column not found",
            )
        
        record_count = len(df)
        
        if record_count < self.min_history_days:
            return ValidationResult(
                check_name="min_history",
                passed=False,
                message=f"Only {record_count} records (need {self.min_history_days})",
                details={
                    "record_count": record_count,
                    "required": self.min_history_days,
                },
                severity="warning",
            )
        
        return ValidationResult(
            check_name="min_history",
            passed=True,
            message=f"{record_count} records (>= {self.min_history_days})",
        )


def validate_data(
    df: pd.DataFrame,
    commodity_id: str = "unknown",
    fail_on_error: bool = True,
    config_path: Optional[str] = None,
) -> ValidationReport:
    """
    Convenience function to validate data.
    
    Args:
        df: DataFrame to validate
        commodity_id: ID for reporting
        fail_on_error: If True, raise exception on validation failure
        config_path: Optional path to config
        
    Returns:
        ValidationReport
        
    Raises:
        ValueError: If fail_on_error=True and validation fails
    """
    validator = DataValidator(config_path)
    report = validator.validate(df, commodity_id)
    
    if fail_on_error and not report.passed:
        raise ValueError(f"Data validation failed:\n{report.summary()}")
    
    return report
