import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt

# loading the model
@st.cache_resource
def load_model():
    return joblib.load("best_price_model.pkl")
model= load_model()

# load clean data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")
df=load_data()

st.title("Real Time Crop Price Forcasting APP")

# dropdowm for crop and market
crop=df["Grade"].unique()
selected_crop=st.selectbox("Selected crop",crop)

if "Market Name" in df.columns:
    markets=df["Market Name"].unique()
    selected_market=st.selectbox("selected market",markets)
else:
    selected_market=None

# forcast day slider
forecast_days=st.slider("Number of days",min_value=5,max_value=30,value=7)


#filter data
filter_df= df[df["Grade"]==selected_crop].copy()
if selected_market:
    filter_df= df[df["Market Name"]==selected_market]

filter_df=filter_df.sort_values("Price Date")




filter_df["Price Date"] = pd.to_datetime(filter_df["Price Date"], errors="coerce")
filter_df = filter_df.dropna(subset=["Price Date"])
filter_df = filter_df.sort_values("Price Date")




# prepare forecasting
history=filter_df.copy()

predictions=[]
# Ensure dates are in datetime format
history["Price Date"] = pd.to_datetime(history["Price Date"], errors="coerce")





# Drop any rows where date couldn't be parsed
history = history.dropna(subset=["Price Date"])


for _ in range(forecast_days):
    # Create lag features
    for lag in [1, 2, 3]:
        history[f"lag_{lag}"] = history["Modal Price (Rs./Quintal)"].shift(lag)
    
    # Create moving averages
    history["ma_3"] = history["Modal Price (Rs./Quintal)"].rolling(window=3).mean()
    history["ma_7"] = history["Modal Price (Rs./Quintal)"].rolling(window=7).mean()
   

    # Drop NaN rows
    features = history.dropna().iloc[-1][['lag_1', 'lag_2', 'lag_3', 'ma_3', 'ma_7']].values.reshape(1, -1)

    # Predict next price
    next_pred = model.predict(features)[0]
    predictions.append(next_pred)

    # Append to history for recursive prediction
    
    next_date = history["Price Date"].max() + pd.Timedelta(days=1)
    history = pd.concat([history, pd.DataFrame({
        "Price Date": [next_date],
        "Modal Price (Rs./Quintal)": [next_pred]
    })], ignore_index=True)

# Prepare chart data
actual_dates = filter_df["Price Date"].tail(30)
actual_prices = filter_df["Modal Price (Rs./Quintal)"].tail(30)

future_dates = pd.date_range(start=filter_df["Price Date"].max() + pd.Timedelta(days=1), periods=forecast_days)
pred_prices = predictions

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_dates, actual_prices, label="Actual Price", marker='o')
ax.plot(future_dates, pred_prices, label="Forecast Price", marker='x')
ax.set_xlabel("Date")
ax.set_ylabel("Price (Rs./Quintal)")
ax.set_title(f"{selected_crop} Price Forecast")
ax.legend()

st.pyplot(fig)

