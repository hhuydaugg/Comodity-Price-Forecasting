# Example: Visualizing Commodity Price Predictions

This notebook demonstrates how to visualize predictions vs ground truth for commodity price forecasting.

## Setup

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.plot_predictions import PredictionVisualizer, plot_predictions_quick
from src.ingestion.loader import CommodityLoader
from src.inference.predictor import BatchPredictor
```

## Generate Predictions

First, let's generate some predictions:

```python
# Initialize predictor
predictor = BatchPredictor()

# Generate predictions for crude oil
predictions = predictor.predict_commodity("crude_oil", horizons=[1, 7, 30])

# Display predictions
print(predictions)
```

## Visualization Option 1: Quick Plot

The simplest way to visualize predictions:

```python
# If you have saved predictions to a file
plot_predictions_quick(
    commodity_id="crude_oil",
    predictions_path="data/predictions/predictions_20240210_120000.parquet",
    lookback_days=90,
    save_path="figures/crude_oil_predictions.png"
)
plt.show()
```

## Visualization Option 2: Detailed Plotting

For more control, use the `PredictionVisualizer` class:

```python
# Initialize visualizer
viz = PredictionVisualizer()

# Load actual data
loader = CommodityLoader()
actual_data = loader.load_commodity("crude_oil")

# Plot predictions vs actual
fig = viz.plot_prediction_vs_actual(
    commodity_id="crude_oil",
    predictions_df=predictions,
    actual_df=actual_data,
    lookback_days=90,
    save_path="figures/crude_oil_forecast.png"
)
plt.show()
```

## Multi-Horizon Comparison

Compare different forecast horizons side-by-side:

```python
fig = viz.plot_multi_horizon_comparison(
    commodity_id="crude_oil",
    predictions_df=predictions,
    actual_df=actual_data,
    save_path="figures/crude_oil_multi_horizon.png"
)
plt.show()
```

## Forecast Accuracy Analysis

Analyze prediction accuracy across different horizons:

```python
# Note: This requires actual data at the target dates
fig = viz.plot_forecast_accuracy(
    predictions_df=predictions,
    actual_df=actual_data,
    save_path="figures/forecast_accuracy.png"
)
plt.show()
```

## Complete Example

Here's a complete workflow:

```python
from src.visualization.plot_predictions import PredictionVisualizer
from src.ingestion.loader import CommodityLoader
from src.inference.predictor import BatchPredictor

# 1. Generate predictions
predictor = BatchPredictor()
predictions = predictor.predict_commodity("crude_oil", horizons=[1, 7, 30])

# 2. Load actual data
loader = CommodityLoader()
actual_data = loader.load_commodity("crude_oil")

# 3. Create visualizations
viz = PredictionVisualizer()

# Main prediction plot
fig1 = viz.plot_prediction_vs_actual(
    commodity_id="crude_oil",
    predictions_df=predictions,
    actual_df=actual_data,
    lookback_days=60,
    figsize=(14, 6)
)

# Multi-horizon comparison
fig2 = viz.plot_multi_horizon_comparison(
    commodity_id="crude_oil",
    predictions_df=predictions,
    actual_df=actual_data,
    figsize=(14, 8)
)

plt.show()
```

## Customization

You can customize colors and styles:

```python
# Custom color scheme
viz = PredictionVisualizer(style="seaborn-v0_8-whitegrid")
viz.colors["actual"] = "#1f77b4"
viz.colors["prediction"] = "#ff7f0e"

# Create plot with custom settings
fig = viz.plot_prediction_vs_actual(
    commodity_id="crude_oil",
    predictions_df=predictions,
    lookback_days=120,
    figsize=(16, 7)
)
```

## Command Line Usage

You can also use the visualization script from the command line:

```bash
python -m src.visualization.plot_predictions crude_oil data/predictions/predictions.parquet figures/output.png
```
