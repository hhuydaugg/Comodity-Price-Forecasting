import json
import os

nb_path = "d:/Work/Comodity-Price-Forecasting/notebook_demo_v2.ipynb"

try:
    print(f"Reading notebook from {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # New code for the function cell
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
    last_row = current_df.iloc[-1]
    static_data = {c: last_row[c] for c in static_cols if c in current_df.columns}
    
    last_date = pd.to_datetime(current_df['date'].iloc[-1])
    
    # Generate business days for future
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    
    print(f"Predicting {horizon} steps into the future...")
    
    target_col = feature_generator.target_column
    
    for i, date in enumerate(future_dates):
        # 1. Create a new row for the next date
        row_data = {'date': [date]}
        # Propagate static data to ensure grouping works correctly in feature generation
        for k, v in static_data.items():
            row_data[k] = [v]
            
        new_row = pd.DataFrame(row_data)
        
        # 2. Append to current dataframe (with NaNs for other columns)
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # 3. Regenerate features
        # Note: This is computationally expensive but ensures correctness for all rolling/lag features
        current_df_feat = feature_generator.generate(current_df)
        
        # 4. Prepare input for model
        if not hasattr(model, 'feature_names_'):
             # Fallback if feature_names_ not set
             exclude = {'date', target_col, 'commodity', 'commodity_id'}
             feats = [c for c in current_df_feat.columns if c not in exclude and pd.api.types.is_numeric_dtype(current_df_feat[c])]
        else:
             feats = model.feature_names_
             
        # 5. Predict next value
        try:
            # Most models here have seq_len ~ 60
            window_size = 100 # ample buffer
            input_df = current_df_feat.tail(window_size)[feats]
            
            # Fill NaNs in features if any (important for the newly added row features that depend on lag)
            input_df = input_df.fillna(method='ffill').fillna(0)
            
            # Predict
            pred_array = model.predict(horizon=1, X=input_df)
            
            # Handle different return shapes just in case
            if hasattr(pred_array, 'item'):
                pred_value = pred_array.item()
            else:
                pred_value = float(pred_array[-1])
            
        except Exception as e:
            print(f"Prediction failed at step {i}: {e}")
            break
        
        # 6. Update target in dataframe for next iteration
        current_df.loc[current_df.index[-1], target_col] = pred_value
        
        if 'close' in current_df.columns and target_col != 'close':
             current_df.loc[current_df.index[-1], 'close'] = pred_value
        
        future_preds.append({'date': date, 'pred': pred_value})

    return pd.DataFrame(future_preds)"""

    # New code for the execution cell
    new_execution_code = """# --- Run Future Predictions ---
FUTURE_STEPS = 30
future_forecasts = {}

print(f"Generating {FUTURE_STEPS} days forecast for each model...")

# Ensure df_clean (with history) is used as starter
# Use the FULL df_clean to ensure all columns (like z_score, missing_streak) are present
if 'df_clean' in locals() and 'fg' in locals():
    # Make a clean starter df with ALL columns
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
            print(f"Failed to forecast: {e}")
            # import traceback
            # traceback.print_exc()
            
    print("\\nDone!")
else:
    print("Warning: df_clean or fg not found in locals. Make sure previous cells are run.")"""

    # Helper to convert string to source lines
    def to_source(code_str):
        lines = [line + '\n' for line in code_str.split('\n')]
        if lines and lines[-1].endswith('\n'):
             lines[-1] = lines[-1][:-1]
        return lines

    # Find and update cells
    found_func = False
    found_exec = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'def predict_future_autoregressive' in source:
                cell['source'] = to_source(new_function_code)
                found_func = True
                print("Updated function cell.")
            elif 'FUTURE_STEPS =' in source and 'starter_df =' in source:
                cell['source'] = to_source(new_execution_code)
                found_exec = True
                print("Updated execution cell.")

    if found_func and found_exec:
        print(f"Writing updated notebook to {nb_path}...")
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print(f"Could not find cells to update. Func found: {found_func}, Exec found: {found_exec}")

except Exception as e:
    print(f"Error: {e}")
