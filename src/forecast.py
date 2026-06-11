import os
import joblib
import datetime
import pandas as pd
import numpy as np
from src.features import get_season

# Define paths
MODELS_DIR = "models"
INPUT_FILE = "data/processed/cleaned_master.csv"

def forecast_next_n_days(crop, state, district, market, forecast_days=7):
    """
    Recursively forecasts the modal price for the next N days.
    """
    model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train models first.")
        
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cleaned dataset not found at {INPUT_FILE}. Please clean data first.")
        
    # Load the best pipeline (contains ColumnTransformer preprocessor + Regressor)
    pipeline = joblib.load(model_path)
    
    # Load master dataset
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # Filter for the specific time-series key
    series_df = df[
        (df["Crop"] == crop) &
        (df["State"] == state) &
        (df["District Name"] == district) &
        (df["Market Name"] == market)
    ].sort_values("date").copy()
    
    if len(series_df) < 10:
        # Fallback if too few records are found for this specific market combination
        # We search for the crop generally
        print(f"[!] Warning: Too few records for {crop} in {market}. Using general crop fallback.")
        series_df = df[df["Crop"] == crop].sort_values("date").copy()
        if len(series_df) == 0:
            raise ValueError(f"No records found for crop: {crop} in dataset.")

    # We make a copy of history to append our forecasts recursively
    history = series_df.copy()
    forecasts = []
    
    for _ in range(forecast_days):
        # 1. Get the last date in history
        last_date = history["date"].max()
        next_date = last_date + datetime.timedelta(days=1)
        
        # 2. Extract calendar features
        day = next_date.day
        month = next_date.month
        year = next_date.year
        week_number = next_date.isocalendar()[1]
        season = get_season(month)
        
        # 3. Extract lag features (from the last rows in history)
        # lag_1 is the price of today (which is the last row in history)
        last_rows = history.tail(10).copy()
        
        lag_1 = float(last_rows.iloc[-1]["modal_price"])
        lag_2 = float(last_rows.iloc[-2]["modal_price"]) if len(last_rows) >= 2 else lag_1
        lag_7 = float(last_rows.iloc[-7]["modal_price"]) if len(last_rows) >= 7 else lag_1
        
        # 4. Extract rolling average features
        ma_3 = last_rows.tail(3)["modal_price"].mean()
        ma_7 = last_rows.tail(7)["modal_price"].mean()
        
        # 5. Extract daily return/change features
        price_change_pct = (lag_1 - lag_2) / (lag_2 + 1e-5)
        
        # 6. Extract price volatility (std of returns over last 7 days)
        # Calculate daily change percentage for history tails to compute std
        history_tail = history.tail(8).copy()
        history_tail["lag_price"] = history_tail["modal_price"].shift(1)
        history_tail["pct_chg"] = (history_tail["modal_price"] - history_tail["lag_price"]) / (history_tail["lag_price"] + 1e-5)
        price_volatility = history_tail["pct_chg"].tail(7).std()
        
        # Fill NaN volatility
        if pd.isna(price_volatility):
            price_volatility = 0.0
            
        # Build features DataFrame matching train.py features list
        features_dict = {
            "day": [day],
            "month": [month],
            "year": [year],
            "week_number": [week_number],
            "season": [season],
            "State": [state],
            "District Name": [district],
            "Market Name": [market],
            "lag_1": [lag_1],
            "lag_2": [lag_2],
            "lag_7": [lag_7],
            "ma_3": [ma_3],
            "ma_7": [ma_7],
            "price_change_pct": [price_change_pct],
            "price_volatility": [price_volatility]
        }
        X_next = pd.DataFrame(features_dict)
        
        # 7. Predict next price
        pred_price = float(pipeline.predict(X_next)[0])
        # Ensure predicted price is positive
        pred_price = max(100.0, pred_price)
        
        # 8. Append to history for the next iteration
        new_row = {
            "date": next_date,
            "Crop": crop,
            "State": state,
            "District Name": district,
            "Market Name": market,
            "min_price": pred_price * 0.9,
            "max_price": pred_price * 1.1,
            "modal_price": pred_price
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        
        forecasts.append({
            "date": next_date,
            "predicted_price": int(pred_price)
        })
        
    return pd.DataFrame(forecasts)
