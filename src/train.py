import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Define paths
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
INPUT_FILE = os.path.join(PROCESSED_DIR, "featured_master.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define feature sets
NUMERIC_FEATURES = [
    "day", "month", "year", "week_number",
    "lag_1", "lag_2", "lag_7",
    "ma_3", "ma_7",
    "price_change_pct", "price_volatility"
]
CATEGORICAL_FEATURES = ["season", "State", "District Name", "Market Name"]
TARGET = "modal_price"

def evaluate_model(y_true, y_pred):
    """
    Computes standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def train_crop_model(crop_name, crop_df):
    """
    Trains and compares ML models for a specific crop and saves the best one.
    """
    print(f"\n" + "-"*50)
    print(f"[*] Training Models for Crop: {crop_name}")
    print(f"[-] Total records: {len(crop_df)}")
    
    # Sort chronologically to preserve time sequence
    crop_df = crop_df.sort_values("date")
    
    # Split features and target
    X = crop_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = crop_df[TARGET]
    
    # Temporal split: 80% train, 20% test to avoid future leakage
    split_idx = int(len(crop_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[-] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 1. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    
    # 2. Define Candidate Models & Grid Parameters
    # Time-Series Split is critical for sequential data validation (prevents shuffling)
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [100, 150],
                "regressor__max_depth": [5, 10, None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [50, 100],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5]
            }
        }
    }
    
    best_crop_model = None
    best_mae = float("inf")
    results = []
    
    # 3. Model Search & CV Comparison
    for name, config in models.items():
        # Build scikit-learn Pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", config["model"])
        ])
        
        print(f"[*] Training {name}...")
        
        if config["params"]:
            # Perform GridSearchCV utilizing TimeSeriesSplit validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid=config["params"],
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            cv_score = -grid_search.best_score_
            print(f"    Best Params: {grid_search.best_params_}")
            print(f"    5-Split CV MAE: {cv_score:.2f}")
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            cv_score = np.nan  # LR doesn't run CV tuning
            
        # Evaluate on Holdout Test Set
        y_pred = best_pipeline.predict(X_test)
        mae, rmse, r2 = evaluate_model(y_test, y_pred)
        
        results.append({
            "Model": name,
            "CV MAE": f"{cv_score:.2f}" if not np.isnan(cv_score) else "N/A",
            "Test MAE": mae,
            "Test RMSE": rmse,
            "Test R2": r2
        })
        
        print(f"    Test Set Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        
        # Save model if it has the best test MAE
        if mae < best_mae:
            best_mae = mae
            best_crop_model = best_pipeline
            best_model_name = name

    # 4. Print Comparison Table for Crop
    results_df = pd.DataFrame(results)
    print(f"\n--- Model Comparison Summary for {crop_name} ---")
    print(results_df.to_string(index=False))
    
    # 5. Save best performing pipeline to disk
    model_save_path = os.path.join(MODELS_DIR, f"{crop_name.lower()}_model.pkl")
    # Save the entire pipeline including preprocessing ColumnTransformer
    joblib.dump(best_crop_model, model_save_path)
    print(f"\n[+] Selected {best_model_name} as the best model for {crop_name}.")
    print(f"[+] Saved model pipeline package to: {model_save_path}")
    print("-"*50)
    
    return results

def main():
    print("=" * 60)
    print(" Smart Agricultural Price Prediction System - Model Training")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"[-] Input file not found: {INPUT_FILE}. Please run feature engineering first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    crops = df["Crop"].unique()
    all_results = {}
    
    for crop in crops:
        crop_df = df[df["Crop"] == crop]
        crop_results = train_crop_model(crop, crop_df)
        all_results[crop] = crop_results
        
    print("\n[+] Model building process completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
