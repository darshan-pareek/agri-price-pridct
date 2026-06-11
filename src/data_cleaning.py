import os
import pandas as pd
import numpy as np

# Define file paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MASTER_FILE = os.path.join(PROCESSED_DIR, "cleaned_master.csv")

FILE_MAPPING = {
    "Onion": os.path.join(RAW_DIR, "onion.csv"),
    "Tomato": os.path.join(RAW_DIR, "tamato.csv"),  # using existing file naming
    "Potato": os.path.join(RAW_DIR, "potato.csv")
}

DISTRICT_TO_STATE = {
    "Jaipur": "Rajasthan",
    "Pune": "Maharashtra",
    "Agra": "Uttar Pradesh"
}

def clean_crop_data(crop_name, file_path):
    """
    Cleans a single raw crop CSV file.
    """
    if not os.path.exists(file_path):
        print(f"[-] Raw file not found: {file_path}")
        return pd.DataFrame()

    # Load raw file
    df = pd.read_csv(file_path)
    print(f"[*] Cleaning {crop_name} raw data. Initial shape: {df.shape}")

    # 1. Handle Column Naming
    # Standardize column names for ease of coding
    df.columns = df.columns.str.strip()
    
    # 2. Add Crop Column
    df["Crop"] = crop_name

    # 3. Standardize and parse Date column
    # Date in raw data has format DD-MMM-YY (e.g. 22-May-25) or YYYY-MM-DD
    # We will try standard pandas datetime parsing
    df["date"] = pd.to_datetime(df["Price Date"], errors="coerce")
    
    # 4. Standardize and parse Price columns
    price_cols = {
        "Min Price (Rs./Quintal)": "min_price",
        "Max Price (Rs./Quintal)": "max_price",
        "Modal Price (Rs./Quintal)": "modal_price"
    }
    
    # If the file uses the renamed columns (from API), rename them back
    for old_col, new_col in price_cols.items():
        if old_col in df.columns:
            df[new_col] = pd.to_numeric(df[old_col], errors="coerce")
        elif new_col in df.columns:
            df[new_col] = pd.to_numeric(df[new_col], errors="coerce")
            
    # Drop rows where crucial variables like date or modal price are missing
    df = df.dropna(subset=["date", "modal_price"])

    # 5. Populate State column based on District Name
    # Raw historical records do not have a State column, but new API/simulated ones do
    if "State" not in df.columns:
        df["State"] = df["District Name"].map(DISTRICT_TO_STATE)
    
    # Fill remaining missing states
    df["State"] = df["State"].fillna(df["District Name"].map(DISTRICT_TO_STATE))
    # Set fallback if district is unknown
    df["State"] = df["State"].fillna("Rajasthan") 

    # Clean text columns
    df["District Name"] = df["District Name"].str.strip()
    df["Market Name"] = df["Market Name"].str.strip()

    # 6. Duplicate Analysis and Removal
    # Sort by date so we keep the latest record if duplicates exist
    df = df.sort_values("date")
    duplicate_count = df.duplicated(subset=["date", "Crop", "State", "District Name", "Market Name"]).sum()
    if duplicate_count > 0:
        print(f"[!] Found {duplicate_count} duplicate rows in {crop_name}. Removing...")
        df = df.drop_duplicates(subset=["date", "Crop", "State", "District Name", "Market Name"], keep="last")

    # 7. Outlier Detection using Interquartile Range (IQR)
    # We calculate IQR for the modal_price to remove extreme data entry errors
    q25 = df["modal_price"].quantile(0.25)
    q75 = df["modal_price"].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    # We apply a safety minimum price of 100 Rs/Quintal (as prices cannot be free)
    lower_bound = max(100.0, lower_bound)
    
    outliers = df[(df["modal_price"] < lower_bound) | (df["modal_price"] > upper_bound)]
    if not outliers.empty:
        print(f"[!] Identified {len(outliers)} price outliers in {crop_name} (IQR bounds: {lower_bound:.2f} - {upper_bound:.2f}). Filtering...")
        df = df[(df["modal_price"] >= lower_bound) & (df["modal_price"] <= upper_bound)]

    # Select and order final cleaned columns
    final_cols = ["date", "Crop", "State", "District Name", "Market Name", "min_price", "max_price", "modal_price"]
    df = df[final_cols]
    
    print(f"[+] Cleaned {crop_name} data. Final shape: {df.shape}")
    return df

def clean_and_merge_all():
    """
    Cleans all crop data files and merges them into a single master CSV.
    """
    print("=" * 60)
    print(" Smart Agricultural Price Prediction System - Data Cleaning")
    print("=" * 60)
    
    cleaned_dfs = []
    for crop, file_path in FILE_MAPPING.items():
        cleaned_df = clean_crop_data(crop, file_path)
        if not cleaned_df.empty:
            cleaned_dfs.append(cleaned_df)
            
    if cleaned_dfs:
        master_df = pd.concat(cleaned_dfs, ignore_index=True)
        # Final sort by date
        master_df = master_df.sort_values("date")
        
        # Save to processed directory
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        master_df.to_csv(MASTER_FILE, index=False)
        print(f"\n[+] Merged master file saved to: {MASTER_FILE}")
        print(f"[+] Total records in Master dataset: {master_df.shape[0]}")
        print(f"[+] Crops present: {master_df['Crop'].value_counts().to_dict()}")
        print(f"[+] States represented: {master_df['State'].value_counts().to_dict()}")
    else:
        print("[-] Error: No data was cleaned. Master file not created.")
    print("=" * 60)

if __name__ == "__main__":
    clean_and_merge_all()
