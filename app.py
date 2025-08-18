import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_master.csv", parse_dates=["date"])
    return df

def load_model(crop):
    return joblib.load(f"{crop.lower()}_model.pkl")

df = load_data()

st.title("🌾 Multi-Crop Price Forecasting App")

# Dropdowns
crop = st.selectbox("Select Crop", sorted(df["Crop"].unique()))
forecast_days = st.slider("Days to Forecast", 5, 300, 7)

# Filter data
data = df[df["Crop"] == crop].sort_values("date").copy()
model = load_model(crop)

# Forecast
history = data.copy()
predictions = []
dates = []

for _ in range(forecast_days):
    for lag in [1, 2, 3]:
        history[f"lag_{lag}"] = history["modal_price"].shift(lag)
    history["ma_3"] = history["modal_price"].rolling(window=3).mean()
    history["ma_7"] = history["modal_price"].rolling(window=7).mean()
    
    last_row = history.dropna().iloc[-1]
    X_last = last_row[["lag_1", "lag_2", "lag_3", "ma_3", "ma_7"]].values.reshape(1, -1)
    
    pred = model.predict(X_last)[0]
    next_date = history["date"].max() + pd.Timedelta(days=1)
    
    predictions.append(pred)
    dates.append(next_date)
    
    history = pd.concat([history, pd.DataFrame({"date": [next_date], "modal_price": [pred], "Crop": [crop]})])

# Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data["date"], data["modal_price"], label="Actual Price")
ax.plot(dates, predictions, label="Forecast Price", linestyle="--", marker="o")
ax.set_title(f"{crop} Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price (Rs./Quintal)")
ax.legend()
st.pyplot(fig)
