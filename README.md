# Commodity Price Prediction System

Hệ thống dự đoán giá hàng hóa theo chuẩn MLOps, hỗ trợ nhiều mặt hàng với dữ liệu daily.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python -m src.training.trainer --commodity all

# Run batch inference
python -m src.inference.predictor --date today
```

## 📁 Project Structure

```
commodity_forecast/
├── configs/           # Configuration files
├── data/
│   ├── raw/          # Bronze: Immutable raw data
│   ├── processed/    # Silver: Cleaned data
│   └── features/     # Gold: Feature datasets
├── src/
│   ├── ingestion/    # Data loading & validation
│   ├── preprocessing/# Data cleaning
│   ├── features/     # Feature engineering
│   ├── models/       # Model implementations
│   ├── training/     # Training pipeline
│   ├── inference/    # Batch prediction
│   ├── evaluation/   # Metrics & backtesting
│   └── monitoring/   # Drift detection
├── orchestration/    # Airflow DAGs
├── serving/          # FastAPI
├── notebooks/        # EDA & experiments
└── tests/            # Unit & integration tests
```

## 📊 Supported Models

- **Baseline**: Naive, Seasonal Naive
- **Statistical**: ARIMA, ETS
- **ML**: XGBoost, LightGBM

## 🔧 Configuration

Edit `configs/commodities.yaml` to add/modify commodities.
Edit `configs/model_config.yaml` for model hyperparameters.

## 📈 Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MASE (Mean Absolute Scaled Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)

## 🐳 Docker

```bash
docker build -t commodity-forecast .
docker run commodity-forecast python -m src.inference.predictor
```

## 📝 License

MIT
