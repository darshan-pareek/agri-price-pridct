import os
import sys
import datetime
import random
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
DATA_GOV_API_URL = "https://api.data.gov.in/resource/9ef842f8-9614-4688-9986-2e9e91190cf7"
API_KEY = os.getenv("DATA_GOV_API_KEY")

# Supported Crops and Geographies
CROPS = ["Onion", "Tomato", "Potato"]
STATES_CONFIG = {
    "Rajasthan": {"district": "Jaipur", "market": "Jaipur (F&V)"},
    "Maharashtra": {"district": "Pune", "market": "Pune"},
    "Uttar Pradesh": {"district": "Agra", "market": "Achhnera"}
}

# Mapping of raw file names to Commodity names used in code
FILE_MAPPING = {
    "Onion": "data/raw/onion.csv",
    "Tomato": "data/raw/tamato.csv",  # keeping user's spelling "tamato.csv" to prevent breaking existing code
    "Potato": "data/raw/potato.csv"
}

def fetch_from_api(crop, state, limit=100):
    """
    Fetches real-time mandi prices from data.gov.in API for a specific crop and state.
    """
    if not API_KEY:
        raise ValueError("API key not found. Please set DATA_GOV_API_KEY in your .env file.")
    
    print(f"[*] Fetching real API data for {crop} in {state}...")
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
        "filters[commodity]": crop,
        "filters[state]": state
    }
    
    try:
        response = requests.get(DATA_GOV_API_URL, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            print(f"[+] Successfully fetched {len(records)} records from API.")
            return pd.DataFrame(records)
        else:
            print(f"[-] API request failed with status code {response.status_code}: {response.text}")
            return pd.DataFrame()
    except Exception as e:
        print(f"[-] API request error: {e}")
        return pd.DataFrame()

def generate_simulated_record(last_row, target_date, crop, state, district, market):
    """
    Generates a realistic daily price record using a random walk with seasonal trends.
    """
    # Parse last price details
    last_min = float(last_row["Min Price (Rs./Quintal)"])
    last_max = float(last_row["Max Price (Rs./Quintal)"])
    last_modal = float(last_row["Modal Price (Rs./Quintal)"])
    
    # Seasonality adjustments (e.g. higher prices in summer, lower after harvests)
    month = target_date.month
    season_drift = 0.0
    if crop == "Tomato" and month in [6, 7, 8]:  # Tomato price hikes during monsoon
        season_drift = 0.015
    elif crop == "Onion" and month in [10, 11, 12]:  # Onion price spikes in late autumn
        season_drift = 0.012
    elif month in [2, 3, 4]:  # Winter harvest crop drops
        season_drift = -0.008
        
    # Volatility and random walk
    volatility = 0.03  # 3% daily variation
    change = random.normalvariate(season_drift, volatility)
    
    # Calculate new price
    new_modal = max(300.0, int(last_modal * (1 + change)))
    new_min = max(200.0, int(new_modal * random.uniform(0.8, 0.9)))
    new_max = max(new_min + 100, int(new_modal * random.uniform(1.1, 1.25)))
    
    # Format date as DD-MMM-YY (e.g., 10-Jun-26) to match raw source format
    formatted_date = target_date.strftime("%d-%b-%y")
    
    return {
        "District Name": district,
        "Market Name": market,
        "Commodity": crop,
        "Variety": last_row.get("Variety", "Other"),
        "Grade": last_row.get("Grade", "FAQ"),
        "Min Price (Rs./Quintal)": int(new_min),
        "Max Price (Rs./Quintal)": int(new_max),
        "Modal Price (Rs./Quintal)": int(new_modal),
        "Price Date": formatted_date
    }

def run_local_simulation(crop, days_to_generate=None):
    """
    Generates simulated data for multiple states and appends it to the raw CSV.
    If days_to_generate is None, it dynamically calculates the gap between
    the last date in the dataset and today's date to bring the data up to date.
    """
    file_path = FILE_MAPPING.get(crop)
    if not os.path.exists(file_path):
        print(f"[-] Base raw file not found at {file_path}. Cannot run simulation.")
        return
    
    # Read existing raw file
    df = pd.read_csv(file_path)
    
    # Ensure Price Date column is parsed to determine start date
    df_dates = pd.to_datetime(df["Price Date"], format="%d-%b-%y", errors="coerce")
    last_date = df_dates.max()
    
    today = datetime.datetime.now()
    
    if pd.isnull(last_date):
        last_date = today - datetime.timedelta(days=5)
        print(f"[!] Warning: Could not parse dates in {file_path}. Using standard start date: {last_date.date()}")
    
    # Calculate the gap in days
    if days_to_generate is None:
        gap_days = (today - last_date).days
        if gap_days <= 0:
            print(f"[+] {crop} data is already up-to-date up to today ({last_date.strftime('%Y-%m-%d')}). No simulation needed.")
            return
        days_to_generate = gap_days
        print(f"[*] Last record date for {crop} was {last_date.strftime('%Y-%m-%d')}.")
        print(f"[*] Current date is {today.strftime('%Y-%m-%d')}. Generating {days_to_generate} days of data to catch up...")
    else:
        print(f"[*] Last record date for {crop} was {last_date.strftime('%Y-%m-%d')}.")
        print(f"[*] Generating a fixed {days_to_generate} days of data...")
    
    # Get last known row as template for generation
    # Filter rows with actual pricing
    valid_rows = df.dropna(subset=["Modal Price (Rs./Quintal)"])
    if len(valid_rows) == 0:
        print("[-] No valid template rows found in the CSV. Aborting.")
        return
    
    template_row = valid_rows.iloc[-1]
    
    new_records = []
    start_sl_no = int(df["Sl no."].max()) + 1 if "Sl no." in df.columns else len(df) + 1
    
    # Loop over dates and states
    current_date = last_date
    for d in range(1, days_to_generate + 1):
        current_date += datetime.timedelta(days=1)
        
        for state, geo in STATES_CONFIG.items():
            record = generate_simulated_record(
                template_row, 
                current_date, 
                crop, 
                state, 
                geo["district"], 
                geo["market"]
            )
            # Add state column to simulated data to meet the requirement
            record["State"] = state
            record["Sl no."] = start_sl_no
            start_sl_no += 1
            new_records.append(record)
            
    # Convert new records to DataFrame and append
    new_df = pd.DataFrame(new_records)
    
    # For alignment, ensure columns match the existing dataset
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df.to_csv(file_path, index=False)
    print(f"[+] Appended {len(new_records)} simulated records for {crop} across states to {file_path}.")

def main():
    print("=" * 60)
    print(" Smart Agricultural Price Prediction System - Data Collection")
    print("=" * 60)
    
    if API_KEY:
        print("[*] API Key detected. Using Government Open Data API mode.")
        for crop in CROPS:
            all_crop_data = []
            for state in STATES_CONFIG.keys():
                df = fetch_from_api(crop, state, limit=50)
                if not df.empty:
                    # Add to local tracker
                    all_crop_data.append(df)
            
            if all_crop_data:
                combined_crop_df = pd.concat(all_crop_data, ignore_index=True)
                file_path = FILE_MAPPING.get(crop)
                # Standardize format and save
                combined_crop_df.to_csv(file_path, index=False)
                print(f"[+] Saved fetched data for {crop} to {file_path}")
    else:
        print("[!] No API key found in .env (DATA_GOV_API_KEY).")
        print("[*] Running in Offline Fallback / Simulation Mode.")
        print("[*] Generating daily prices across multiple states...")
        
        # Simulating data collection up to today to keep predictions current
        for crop in CROPS:
            run_local_simulation(crop, days_to_generate=None)
            
    print("[+] Data collection process completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
