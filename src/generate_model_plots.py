import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Define paths
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
INPUT_FILE = os.path.join(PROCESSED_DIR, "featured_master.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Preprocessing features definitions
NUMERIC_FEATURES = [
    "day", "month", "year", "week_number",
    "lag_1", "lag_2", "lag_7",
    "ma_3", "ma_7",
    "price_change_pct", "price_volatility"
]
CATEGORICAL_FEATURES = ["season", "State", "District Name", "Market Name"]
TARGET = "modal_price"

def generate_model_plots():
    print("[*] Loading featured dataset for model visualization...")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    crops = df["Crop"].unique()
    
    # 1. Plot Actual vs Predicted prices
    print("[*] Plotting actual vs predicted prices on test sets...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    for idx, crop in enumerate(crops):
        crop_df = df[df["Crop"] == crop].sort_values("date")
        
        # Split inputs
        X = crop_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = crop_df[TARGET]
        
        split_idx = int(len(crop_df) * 0.8)
        test_dates = crop_df["date"].iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Load best model pipeline
        model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_model.pkl")
        pipeline = joblib.load(model_path)
        
        # Predict
        y_pred = pipeline.predict(X.iloc[split_idx:])
        
        # Plotting
        axes[idx].plot(test_dates, y_test, label="Actual Prices", color="black", linewidth=1.5)
        axes[idx].plot(test_dates, y_pred, label="Predicted Prices", color="red", linestyle="--", linewidth=1.5)
        axes[idx].set_title(f"{crop} Prediction Performance on Test Set")
        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Price (Rs./Quintal)")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted_test.png"))
    plt.close()
    print("[+] Saved actual_vs_predicted_test.png")

    # 2. Plot Feature Importances
    print("[*] Plotting feature importances...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for idx, crop in enumerate(crops):
        # Load best model pipeline
        model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_model.pkl")
        pipeline = joblib.load(model_path)
        
        # Extract preprocessor and regressor
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]
        
        # Retrieve feature names
        cat_encoder = preprocessor.named_transformers_["cat"]
        one_hot_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_features = NUMERIC_FEATURES + one_hot_cols
        
        # Get importances dynamically based on the model type
        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            title_suffix = "Feature Importances"
            xlabel_text = "Relative Importance Score"
        elif hasattr(regressor, "coef_"):
            importances = np.abs(regressor.coef_)
            title_suffix = "Absolute Coefficients"
            xlabel_text = "Absolute Coefficient Value"
        else:
            importances = np.zeros(len(all_features))
            title_suffix = "Features"
            xlabel_text = "Score"
        
        # DataFrame
        feat_imp = pd.DataFrame({"Feature": all_features, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)
        
        # Barplot
        sns.barplot(data=feat_imp, y="Feature", x="Importance", ax=axes[idx], palette="viridis")
        axes[idx].set_title(f"Top 10 {title_suffix} for {crop} Model")
        axes[idx].set_xlabel(xlabel_text)
        axes[idx].set_ylabel("Features")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importances.png"))
    plt.close()
    print("[+] Saved feature_importances.png")
    
    print("[+] All model plots generated successfully.")

if __name__ == "__main__":
    generate_model_plots()
