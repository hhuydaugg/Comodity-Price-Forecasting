import json
import os

nb_path = "d:/Work/Comodity-Price-Forecasting/notebook_demo_v2.ipynb"

try:
    print(f"Reading notebook from {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_function_code = """def predict_future_autoregressive(model, start_df, horizon, feature_generator):
    \"\"\"
    Predict future values autoregressively.
    
    Args:
        model: Trained model (must have predict method)
        start_df: DataFrame containing the initial history (must include all necessary lags/windows)
        horizon: Number of steps to predict
        feature_generator: FeatureGenerator instance to update features
        
    Returns:
        DataFrame with future predictions
    \"\"\"
    current_df = start_df.copy()
    future_preds = []
    
    # Get static columns to propagate (e.g., commodity_id)
    static_cols = ['commodity_id', 'commodity']
    if not current_df.empty:
        last_row = current_df.iloc[-1]
        static_data = {c: last_row[c] for c in static_cols if c in current_df.columns}
    else:
        static_data = {}
    
    if not current_df.empty:
        last_date = pd.to_datetime(current_df['date'].iloc[-1])
    else:
        raise ValueError("start_df is empty!")

    # Generate business days for future
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    print(f"Predicting {horizon} steps into the future...")
    
    target_col = feature_generator.target_column
    
    for i, date in enumerate(future_dates):
        # 1. Create a new row for the next date
        row_data = {'date': [date]}
        # Propagate static data
        for k, v in static_data.items():
            row_data[k] = [v]
            
        new_row = pd.DataFrame(row_data)
        
        # 2. Append to current dataframe
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # 3. Regenerate features
        # Note: This is computationally expensive but ensures correctness for all rolling/lag features
        try:
             current_df_feat = feature_generator.generate(current_df)
        except Exception as e:
             # Fallback if generation fails (e.g. strict checks)
             print(f"Feature generation warning at step {i}: {e}")
             current_df_feat = current_df # risky but better than crash?
        
        # 4. Prepare input for model
        if hasattr(model, 'feature_names_'):
             feats = model.feature_names_
        else:
             # Fallback: check columns in current_df_feat
             # If generation failed, this might be raw columns.
             # but usually models have feature_names_ set if using wrappers.
             exclude = {'date', target_col, 'commodity', 'commodity_id'}
             feats = [c for c in current_df_feat.columns if c not in exclude and pd.api.types.is_numeric_dtype(current_df_feat[c])]
             
        # 5. Predict next value
        try:
            window_size = 100 
            # Ensure columns exist
            valid_feats = [f for f in feats if f in current_df_feat.columns]
            if not valid_feats:
                raise ValueError(f"No valid features found. Model expects: {feats[:5]}...")

            input_df = current_df_feat.tail(window_size)[valid_feats]
            
            # Fill NaNs from lag creation/missing generation
            input_df = input_df.fillna(method='ffill').fillna(0)
            
            # Predict
            pred_array = model.predict(horizon=1, X=input_df)
            
            if hasattr(pred_array, 'item'):
                pred_value = pred_array.item()
            else:
                pred_value = float(pred_array[-1])
            
        except Exception as e:
            print(f"Prediction failed at step {i}: {e}")
            break
        
        # 6. Update target
        current_df.loc[current_df.index[-1], target_col] = pred_value
        if 'close' in current_df.columns and target_col != 'close':
             current_df.loc[current_df.index[-1], 'close'] = pred_value
        
        future_preds.append({'date': date, 'pred': pred_value})

    return pd.DataFrame(future_preds)"""

    new_exec_code = """# --- Run Future Predictions ---
FUTURE_STEPS = 30
future_forecasts = {}

print(f"Generating {FUTURE_STEPS} days forecast for each model...")

# Use the FULL df_clean to ensure all columns (like z_score, missing_streak) are present
if 'df_clean' in locals() and 'fg' in locals():
    # Make a clean starter df with ALL columns matching the model training data
    # We should use df_clean.copy() directly as it contains all historical features + raw data
    starter_df = df_clean.copy()
    
    for name, model in trained_models.items():
        print(f"\\n--- Forecasting with {name} ---")
        try:
            forecast_df = predict_future_autoregressive(
                model, 
                starter_df, 
                horizon=FUTURE_STEPS, 
                feature_generator=fg
            )
            future_forecasts[name] = forecast_df
        except Exception as e:
            print(f"Failed to forecast {name}: {e}")
            
    print("\\nDone!")
else:
    print("Warning: df_clean or fg not found. Run previous cells.")"""


    def to_source_list(s):
        return [l + '\n' for l in s.split('\n')]

    updated = False
    
    # Update function cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def predict_future_autoregressive' in source:
                print(f"Updating function cell at index {i}")
                nb['cells'][i]['source'] = to_source_list(new_function_code)
                updated = True
    
    # Update execution cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'future_forecasts = {}' in source:
                print(f"Updating execution cell at index {i}")
                nb['cells'][i]['source'] = to_source_list(new_exec_code)
                updated = True

    if updated:
        print(f"Writing updated notebook to {nb_path}...")
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("No cells matched.")
        
except Exception as e:
    print(e)
