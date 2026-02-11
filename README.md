# Commodity Price Prediction System

Hệ thống dự đoán giá hàng hóa theo chuẩn MLOps, hỗ trợ nhiều mặt hàng với dữ liệu daily.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline (ML models)
python -m src.training.trainer --commodity all

# Run training with Transformer models
python -m src.training.trainer --commodity all --model-type transformer

# Run training with pre-trained foundation models
python -m src.training.trainer --commodity all --model-type pretrained

# Run batch inference
python -m src.inference.predictor --date today
python -m src.inference.predictor --date today --model-type dlinear
```

## 📓 Notebooks
- `notebook_demo_v2.ipynb`: **Advanced** — Transformer models, DLinear, Autoformer, fine-tuning, model comparison
- `notebook_demo.ipynb`: Original ML demo

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
│   ├── features/     # Feature engineering + sequence datasets
│   ├── models/       # Model implementations (ML + Transformer + Pretrained)
│   ├── training/     # Training pipeline + fine-tuning engine
│   ├── inference/    # Batch prediction
│   ├── evaluation/   # Metrics & backtesting
│   └── monitoring/   # Drift detection
├── orchestration/    # Airflow DAGs
├── serving/          # FastAPI
├── notebooks/        # EDA & experiments
└── tests/            # Unit & integration tests
```

## 📊 Supported Models

### Baseline & Statistical
- **Baseline**: Naive, Seasonal Naive
- **Statistical**: ARIMA, ETS

### Machine Learning
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Linear**: ElasticNet
- **Kernel**: SVR

### Deep Learning (Lightweight Transformers)
| Model | Description | Best For |
|-------|------------|----------|
| **PatchTST** | Patch-based attention | Capturing local temporal patterns |
| **DLinear** | Decomposition + Linear | Fast baseline, often beats Transformers |
| **Autoformer** | Auto-Correlation + Decomposition | Periodic/seasonal patterns |
| **iTransformer** | Inverted attention (across features) | Multivariate correlated features |
| **TSTransformer** | Vanilla Transformer encoder | Simple Transformer baseline |

### Foundation Models (Pre-trained)
| Model | Source | Description |
|-------|--------|------------|
| **Chronos** | Amazon | T5-based probabilistic tokenized model |
| **Lag-Llama** | TS Foundation Models | LLM-inspired univariate probabilistic |
| **Moirai** | Salesforce | Universal multi-scale forecaster |
| **Timer** | Tsinghua | Generative pre-trained Transformer |

## 🔧 Fine-tuning

The `TransformerFineTuner` provides production-quality fine-tuning:

```python
from src.training.finetuner import TransformerFineTuner

finetuner = TransformerFineTuner(
    model=model,
    lr=5e-5,
    epochs=10,
    warmup_steps=100,
    grad_accum_steps=2,
    use_amp=True,  # Mixed precision
)
train_loader, val_loader = finetuner.prepare_data(df, feature_cols)
results = finetuner.finetune(train_loader, val_loader)
```

## ⚙️ Configuration

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
