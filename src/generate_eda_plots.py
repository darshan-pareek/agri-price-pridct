import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 15

# Define paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_plots():
    print("[*] Loading datasets for plotting...")
    # Read raw datasets for outlier boxplot
    onion_raw = pd.read_csv(os.path.join(RAW_DIR, "onion.csv"))
    tomato_raw = pd.read_csv(os.path.join(RAW_DIR, "tamato.csv"))
    potato_raw = pd.read_csv(os.path.join(RAW_DIR, "potato.csv"))
    
    # Read cleaned master dataset
    master_df = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_master.csv"))
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df["month"] = master_df["date"].dt.month

    # --- Outlier Boxplot Before Cleaning ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(y=onion_raw["Modal Price (Rs./Quintal)"], ax=axes[0], color="purple").set_title("Raw Onion Modal Prices")
    sns.boxplot(y=tomato_raw["Modal Price (Rs./Quintal)"], ax=axes[1], color="red").set_title("Raw Tomato Modal Prices")
    sns.boxplot(y=potato_raw["Modal Price (Rs./Quintal)"], ax=axes[2], color="gold").set_title("Raw Potato Modal Prices")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "outliers_before_cleaning.png"))
    plt.close()
    print("[+] Saved outliers_before_cleaning.png")

    # --- 1. Line Chart: Price Trends Over Time ---
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=master_df, x="date", y="modal_price", hue="Crop", palette=["purple", "red", "orange"], alpha=0.8)
    plt.title("Mandi Modal Price Trends (2020 - 2026)")
    plt.xlabel("Date")
    plt.ylabel("Price (Rs./Quintal)")
    plt.legend(title="Crops")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_line_chart_price_trends.png"))
    plt.close()
    print("[+] Saved 01_line_chart_price_trends.png")

    # --- 2. Histograms: Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["purple", "red", "orange"]
    crops = ["Onion", "Tomato", "Potato"]

    for idx, crop in enumerate(crops):
        crop_data = master_df[master_df["Crop"] == crop]
        sns.histplot(crop_data["modal_price"], kde=True, ax=axes[idx], color=colors[idx], bins=30)
        axes[idx].set_title(f"{crop} Price Distribution")
        axes[idx].set_xlabel("Modal Price (Rs./Quintal)")
        axes[idx].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_histogram_distributions.png"))
    plt.close()
    print("[+] Saved 02_histogram_distributions.png")

    # --- 3. Boxplot: Seasonality and Spread ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.boxplot(data=master_df, x="Crop", y="modal_price", palette=["purple", "red", "orange"], ax=axes[0])
    axes[0].set_title("Price Spread Comparison by Crop")
    axes[0].set_ylabel("Price (Rs./Quintal)")

    sns.boxplot(data=master_df, x="month", y="modal_price", ax=axes[1], color="skyblue")
    axes[1].set_title("Monthly Seasonality of All Crops")
    axes[1].set_xlabel("Month of Year")
    axes[1].set_ylabel("Price (Rs./Quintal)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_boxplot_seasonality.png"))
    plt.close()
    print("[+] Saved 03_boxplot_seasonality.png")

    # --- 4. Heatmap: Correlation ---
    plt.figure(figsize=(8, 6))
    correlation = master_df[["min_price", "max_price", "modal_price"]].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".4f", square=True)
    plt.title("Correlation Heatmap of Price Fields")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_heatmap_correlation.png"))
    plt.close()
    print("[+] Saved 04_heatmap_correlation.png")

    # --- 5. Trend Chart: Crop Seasonality Line Chart ---
    plt.figure(figsize=(14, 6))
    seasonal_trends = master_df.groupby(["Crop", "month"])["modal_price"].mean().reset_index()
    sns.lineplot(data=seasonal_trends, x="month", y="modal_price", hue="Crop", marker="o", palette=["purple", "red", "orange"])
    plt.title("Crop-wise Seasonal Price Patterns (Monthly Averages)")
    plt.xlabel("Month of Year")
    plt.ylabel("Average Modal Price (Rs./Quintal)")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_trend_chart_seasonality.png"))
    plt.close()
    print("[+] Saved 05_trend_chart_seasonality.png")
    
    print("[+] All EDA plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
