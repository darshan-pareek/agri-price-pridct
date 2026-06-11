# Smart Agricultural Price Prediction System

An end-to-end crop market intelligence and price forecasting platform designed for Indian farmers and consumers. The system collects daily market prices, cleans data outliers, generates time-series lag and rolling statistics, retrains predictive models, and presents them in a multi-page Streamlit web dashboard.

---

## 🏛️ Project Directory Structure

Here is the complete organized structure of the project files, along with details of why each file and directory exists:

```text
Agri Price Predictor/
│
├── venv/                       # Local Python virtual environment (isolated dependencies)
│
├── data/                       # Storage for raw and cleaned datasets
│   ├── raw/                    # Raw historical crop CSV files (as ingested by scraper)
│   │   ├── onion.csv
│   │   ├── tamato.csv          # Tomato raw records (retaining user raw spelling)
│   │   └── potato.csv
│   └── processed/              # Cleaned and feature-engineered datasets
│       ├── cleaned_master.csv  # Merged, sorted, and outlier-filtered dataset
│       └── featured_master.csv # Training-ready dataset with lags & rolling averages
│
├── notebooks/                  # Jupyter Notebooks for interactive data science and diagnostics
│   ├── 01_eda_and_cleaning.ipynb   # Interactive analysis of missing values, duplicates, and outliers
│   ├── 02_feature_engineering.ipynb # Interactive verification of target leakage and lag structures
│   └── 03_model_training.ipynb     # Interactive model search, metrics scoring, and feature importances
│
├── scraper/                    # Ingestion modules
│   └── data_collector.py       # Ingests prices from data.gov.in API (or runs fallback simulator)
│
├── models/                     # Saved ML model pipelines (preprocessors + estimators)
│   ├── onion_model.pkl
│   ├── tomato_model.pkl
│   └── potato_model.pkl
│
├── dashboard/                  # Modules governing the Streamlit multi-page frontend
│   ├── __init__.py             # Declares dashboard directory as a Python package
│   ├── home.py                 # Landing page displaying latest prices and project mission
│   ├── farmer.py               # Farmer dashboard (mandi selectors, forecasts, hold/sell support)
│   ├── consumer.py             # Consumer dashboard (₹/kg tables, buy/wait recommendations)
│   ├── analytics.py            # Reports hub displaying visual plots from notebooks
│   └── about.py                # Technical architecture diagram, tech stack, and interview FAQs
│
├── src/                        # Core codebase scripts
│   ├── data_cleaning.py        # Cleans raw CSV files (resolves schema, maps states, applies IQR)
│   ├── features.py             # Extracts calendar season codes, group-wise shifted lags, and volatility
│   ├── train.py                # Pipeline creator, GridSearchCV, and TimeSeriesSplit cross-validator
│   ├── forecast.py             # Recursive forecaster simulating t+1..t+7 price trajectories
│   ├── generate_eda_plots.py   # Generates figures for interactive reports
│   └── generate_model_plots.py # Generates prediction and feature importance figures
│
├── reports/                    # Exported document files and visual figures
│   └── figures/                # Exported PNG plots (rendered on the analytics dashboard)
│
├── app.py                      # Main Streamlit entrance router (handles navigation)
├── daily_update.py             # Orchestrator script executing end-to-end updates (ingest -> clean -> train)
├── requirements.txt            # Package dependencies manifest
└── README.md                   # Project description and run guide (this file)
```

---

## 🚀 Setup & Execution Guide

Follow these commands to set up the environment and run the system:

### 1. Initialize Virtual Environment & Install Dependencies
Create the environment and install all packages listed in `requirements.txt`:
```powershell
# Create venv
python -m venv venv

# Activate venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline Update
Run the daily update script to simulate data ingestion, clean the new inputs, generate features, and retrain all estimators:
```powershell
python daily_update.py
```

### 3. Run the Streamlit Dashboard
Launch the web-based interactive dashboards:
```powershell
streamlit run app.py
```

---

## 🔧 Machine Learning Pipeline Overview

1. **Ingestion (`scraper/data_collector.py`):** Collects prices from data.gov.in (if a `DATA_GOV_API_KEY` is present in `.env`) or falls back to an offline simulator that appends realistic seasonal pricing increments to the raw records.
2. **Preprocessing (`src/data_cleaning.py`):** Fixes invalid records, populates missing states, removes duplicate entries, and filters extreme prices using IQR thresholds.
3. **Features (`src/features.py`):** Calculates temporal variables, groups data by crop, state, and mandi market, and computes shifted price lags and rolling statistics (avoiding future data leakage).
4. **Modeling (`src/train.py`):** Trains and tunes Linear Regression, Random Forest, and Gradient Boosting Regressors using `TimeSeriesSplit` cross-validation. Saves the best-performing pipeline package to `models/`.
5. **Dashboard (`app.py`):** Routes users to distinct consumer and farmer portals, serving retail cost conversions (₹/kg) and recursive decision-support recommendations.
