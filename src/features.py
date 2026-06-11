import os
import pandas as pd
import numpy as np

# Define file paths
PROCESSED_DIR = "data/processed"
INPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_master.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "featured_master.csv")

def get_season(month):
    """
    Maps a calendar month to its corresponding Indian season.
    - Winter: December, January, February
    - Summer (Pre-monsoon/Zaid): March, April, May
    - Monsoon (Kharif): June, July, August, September
    - Post-Monsoon (Rabi sowing): October, November
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def build_features(df):
    """
    Creates calendar and time-series features grouped by Crop, State, District, and Market.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Ensure date is parsed
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Calendar Features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["season"] = df["month"].map(get_season)
    
    # Sort dataset chronologically before rolling/lag operations
    df = df.sort_values("date")
    
    # Grouping keys to isolate individual time-series
    group_cols = ["Crop", "State", "District Name", "Market Name"]
    grouped = df.groupby(group_cols)
    
    # 2. Lag Features (autoregressive terms)
    # lag_1 is the price of the previous available trading day
    df["lag_1"] = grouped["modal_price"].shift(1)
    df["lag_2"] = grouped["modal_price"].shift(2)
    df["lag_7"] = grouped["modal_price"].shift(7)
    
    # 3. Rolling Average Features
    # Note: We shift by 1 before calculating the rolling mean. 
    # This prevents 'data leakage' because we want to predict the price of today (t) 
    # using rolling averages of yesterday (t-1) and earlier.
    df["ma_3"] = grouped["modal_price"].transform(lambda x: x.shift(1).rolling(window=3).mean())
    df["ma_7"] = grouped["modal_price"].transform(lambda x: x.shift(1).rolling(window=7).mean())
    
    # 4. Price Change Percentage (daily return equivalent)
    # Measures the day-over-day price growth rate
    df["price_change_pct"] = (df["lag_1"] - df["lag_2"]) / (df["lag_2"] + 1e-5)
    
    # 5. Price Volatility
    # Standard deviation of price changes over the last 7 available days
    df["price_volatility"] = df.groupby(group_cols)["price_change_pct"].transform(lambda x: x.rolling(window=7).std())
    
    return df

def generate_featured_data():
    print("=" * 60)
    print(" Smart Agricultural Price Prediction System - Feature Engineering")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"[-] Input file not found: {INPUT_FILE}. Please run cleaning first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"[*] Initial dataset size: {df.shape}")
    
    # Build features
    df_featured = build_features(df)
    
    # Check NaN count before dropping
    nan_counts = df_featured.isnull().sum()
    print("\n[*] Missing values introduced by lag/rolling features:")
    print(nan_counts[nan_counts > 0])
    
    # Drop rows that contain NaNs due to rolling/lag windows
    # Since we need at least 7 lag records to populate lag_7 and rolling std,
    # the first 7 records of each unique series will be dropped.
    df_featured = df_featured.dropna()
    print(f"\n[+] Dataset size after dropping lag NaNs: {df_featured.shape}")
    
    # Save output
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_featured.to_csv(OUTPUT_FILE, index=False)
    print(f"[+] Featured master file saved to: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    generate_featured_data()
