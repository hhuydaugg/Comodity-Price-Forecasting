"""
Unit tests for data ingestion module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


class TestCommodityLoader:
    """Tests for CommodityLoader class."""
    
    def test_load_commodity_from_csv(self, tmp_path):
        """Test loading data from CSV file."""
        # Create sample CSV
        csv_content = """date,close
2024-01-01,100.0
2024-01-02,101.5
2024-01-03,102.0
"""
        csv_file = tmp_path / "test_commodity.csv"
        csv_file.write_text(csv_content)
        
        # Create config
        config_content = f"""
commodities:
  - id: test
    name: Test Commodity
    source: file
    file_path: {csv_file}
schema:
  date_column: date
  close_column: close
  date_format: "%Y-%m-%d"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        # Test loading
        from src.ingestion.loader import CommodityLoader
        loader = CommodityLoader(str(config_file))
        df = loader.load_commodity("test")
        
        assert len(df) == 3
        assert "date" in df.columns
        assert "close" in df.columns
        assert "commodity_id" in df.columns
        assert df["commodity_id"].iloc[0] == "test"
    
    def test_standardize_schema(self):
        """Test schema standardization."""
        from src.ingestion.loader import CommodityLoader
        
        # Create DataFrame with different column names
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [100.0, 101.0],
        })
        
        loader = CommodityLoader()
        result = loader._standardize_schema(df, "test")
        
        assert "date" in result.columns
        assert "close" in result.columns
        assert "commodity_id" in result.columns


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        from src.ingestion.validator import DataValidator
        
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=300, freq="D"),
            "close": np.random.uniform(50, 150, 300),
        })
        
        validator = DataValidator()
        report = validator.validate(df, "test")
        
        assert report.passed or report.error_count == 0
    
    def test_validate_missing_columns(self):
        """Test validation fails for missing columns."""
        from src.ingestion.validator import DataValidator
        
        df = pd.DataFrame({
            "something_else": [1, 2, 3],
        })
        
        validator = DataValidator()
        report = validator.validate(df, "test")
        
        assert not report.passed
    
    def test_validate_duplicate_dates(self):
        """Test validation catches duplicate dates."""
        from src.ingestion.validator import DataValidator
        
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "close": [100.0, 101.0, 102.0],
        })
        
        validator = DataValidator()
        report = validator.validate(df, "test")
        
        # Should have a failed check for date uniqueness
        date_unique_result = next(
            (r for r in report.results if r.check_name == "date_unique"),
            None
        )
        assert date_unique_result is not None
        assert not date_unique_result.passed
    
    def test_validate_negative_close(self):
        """Test validation catches negative close values."""
        from src.ingestion.validator import DataValidator
        
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": [100, 101, -50, 102, 103, 104, 105, 106, 107, 108],
        })
        
        validator = DataValidator()
        report = validator.validate(df, "test")
        
        close_positive_result = next(
            (r for r in report.results if r.check_name == "close_positive"),
            None
        )
        assert close_positive_result is not None
        assert not close_positive_result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
