import os
import sys

# Ensure root directory is in python search path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.data_collector import main as run_data_collection
from src.data_cleaning import clean_and_merge_all as run_data_cleaning
from src.features import generate_featured_data as run_feature_engineering
from src.train import main as run_model_training

def run_pipeline():
    print("=" * 70)
    print("  Smart Agricultural Price Prediction System - End-to-End Update Pipeline")
    print("=" * 70)
    
    # Step 1: Collect latest crop mandi prices
    print("\n[STEP 1/4] Running Ingestion Pipeline...")
    run_data_collection()
    
    # Step 2: Clean and merge raw logs
    print("\n[STEP 2/4] Running Preprocessing & Merging...")
    run_data_cleaning()
    
    # Step 3: Rebuild features and lags
    print("\n[STEP 3/4] Rebuilding Time-Series Features...")
    run_feature_engineering()
    
    # Step 4: Retrain regressor pipelines
    print("\n[STEP 4/4] Retraining Machine Learning Models...")
    run_model_training()
    
    print("\n" + "=" * 70)
    print("[+] SUCCESS: End-to-end Daily Update Pipeline completed successfully!")
    print("[+] All models updated and saved.")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
