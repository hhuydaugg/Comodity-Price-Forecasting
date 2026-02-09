# Commodity Price Prediction System - Walkthrough

## 🎯 Overview

Đã triển khai **MVP hoàn chỉnh** cho hệ thống dự đoán giá hàng hóa theo blueprint cung cấp, bao gồm:
- Data pipeline (ingestion → validation → preprocessing → features)
- Multiple model types (baseline, statistical, ML)
- Walk-forward backtesting
- MLflow experiment tracking
- Monitoring & drift detection

---

## 📁 Project Structure

```
d:\Work\commodity_forecast\
├── configs/
│   ├── commodities.yaml      # Commodity definitions
│   └── model_config.yaml     # Model hyperparameters
├── data/
│   └── raw/crude_oil.csv     # Sample data (1 year)
├── src/
│   ├── ingestion/
│   │   ├── loader.py         # CommodityLoader class
│   │   └── validator.py      # DataValidator với 10 checks
│   ├── preprocessing/
│   │   ├── cleaner.py        # DataCleaner (frequency align, missing fill)
│   │   └── transformer.py    # TargetTransformer (price ↔ returns)
│   ├── features/
│   │   └── generator.py      # FeatureGenerator (lag/rolling/calendar)
│   ├── models/
│   │   ├── baseline.py       # Naive, SeasonalNaive, Drift, Mean
│   │   ├── statistical.py    # ARIMA, ETS, Theta
│   │   └── ml.py             # XGBoost, LightGBM, RandomForest
│   ├── evaluation/
│   │   ├── metrics.py        # MAE, RMSE, MAPE, MASE, sMAPE
│   │   └── backtest.py       # TimeSeriesBacktest (walk-forward)
│   ├── training/
│   │   └── trainer.py        # Trainer orchestrator + MLflow
│   ├── inference/
│   │   └── predictor.py      # BatchPredictor for production
│   └── monitoring/
│       └── monitor.py        # DriftDetector, PerformanceMonitor
├── tests/unit/
│   └── test_ingestion.py     # Unit tests
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd d:\Work\commodity_forecast
pip install -r requirements.txt
```

### 2. Train a Model
```bash
python -m src.training.trainer --commodity crude_oil --model-type ml
```

### 3. Run Predictions
```bash
python -m src.inference.predictor --commodity crude_oil --horizon 1 7 30
```

### 4. Run Tests
```bash
pytest tests/ -v
```

---

## ✅ What Was Built

| Component | Files | Description |
|-----------|-------|-------------|
| **Data Ingestion** | `loader.py`, `validator.py` | Load CSV/Parquet, validate với 10 quality checks |
| **Preprocessing** | `cleaner.py`, `transformer.py` | Frequency alignment, missing fill, target transform |
| **Features** | `generator.py` | Lag (t-1→t-30), rolling stats, calendar, volatility |
| **Models** | `baseline.py`, `statistical.py`, `ml.py` | Naive → ARIMA → XGBoost/LightGBM |
| **Evaluation** | `metrics.py`, `backtest.py` | MAE/RMSE/MASE, walk-forward validation |
| **Training** | `trainer.py` | Full pipeline orchestration + MLflow |
| **Inference** | `predictor.py` | Batch predictions với multi-horizon |
| **Monitoring** | `monitor.py` | PSI drift detection, performance tracking |

---

## 🔧 Configuration

### Edit Commodities (`configs/commodities.yaml`)
```yaml
commodities:
  - id: crude_oil
    name: "Crude Oil (WTI)"
    file_path: data/raw/crude_oil.csv
```

### Edit Model Params (`configs/model_config.yaml`)
```yaml
forecast:
  horizons: [1, 7, 30]
models:
  xgboost:
    enabled: true
    params:
      n_estimators: 500
```

---

## 📋 Next Steps

1. **Add real data**: Đặt CSV files vào `data/raw/`
2. **Train models**: Chạy `trainer.py` cho từng commodity
3. **Setup Airflow**: Tạo DAG cho batch daily (optional)
4. **Deploy API**: Implement FastAPI serving (optional)
