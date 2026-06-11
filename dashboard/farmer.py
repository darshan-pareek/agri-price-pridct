import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.forecast import forecast_next_n_days

INPUT_FILE = "data/processed/cleaned_master.csv"

def render():
    st.markdown("""
        <style>
        .farmer-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #047857; /* Green for farmers */
            margin-bottom: 1.5rem;
        }
        .metric-box {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            text-align: center;
        }
        .metric-title {
            font-size: 0.95rem;
            color: #065f46;
            font-weight: 600;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #047857;
            margin: 0.5rem 0;
        }
        .metric-trend {
            font-size: 1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="farmer-title">👩‍🌾 Farmer Market Intelligence Portal</div>', unsafe_allow_html=True)
    st.write("This dashboard is designed specifically for farmers to check current mandi prices and foresee weekly market trends to plan harvests.")

    if not os.path.exists(INPUT_FILE):
        st.error("⚠️ Cleaned dataset not found. Please execute the update pipeline first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # --- Side-by-Side Filtering Dropdowns ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop_options = sorted(df["Crop"].unique())
        selected_crop = st.selectbox("Select Crop", crop_options)
        
    # Filter df by crop
    crop_df = df[df["Crop"] == selected_crop]
    
    with col2:
        state_options = sorted(crop_df["State"].unique())
        selected_state = st.selectbox("Select State", state_options)
        
    # Filter df by crop and state
    state_df = crop_df[crop_df["State"] == selected_state]
    
    with col3:
        # Get unique combinations of District and Market
        district_markets = state_df.groupby(["District Name", "Market Name"]).size().reset_index()
        market_labels = [f"{row['District Name']} - {row['Market Name']}" for idx, row in district_markets.iterrows()]
        
        selected_market_label = st.selectbox("Select Mandi Market", market_labels)
        
        # Extract selected district and market
        selected_idx = market_labels.index(selected_market_label)
        selected_district = district_markets.iloc[selected_idx]["District Name"]
        selected_market = district_markets.iloc[selected_idx]["Market Name"]
        
    # Get history for the specific market
    market_df = state_df[
        (state_df["District Name"] == selected_district) & 
        (state_df["Market Name"] == selected_market)
    ].sort_values("date")
    
    if market_df.empty:
        st.warning("No historical pricing found for this mandi combination.")
        return
        
    latest_record = market_df.iloc[-1]
    current_price = int(latest_record["modal_price"])
    last_date = latest_record["date"]
    
    # Run Forecast for the next 7 days
    with st.spinner("Calculating price forecast..."):
        try:
            forecast_df = forecast_next_n_days(
                selected_crop, 
                selected_state, 
                selected_district, 
                selected_market, 
                forecast_days=7
            )
            next_day_price = forecast_df.iloc[0]["predicted_price"]
            price_in_a_week = forecast_df.iloc[-1]["predicted_price"]
            
            # Calculate weekly change
            pct_change = ((price_in_a_week - current_price) / current_price) * 100
        except Exception as e:
            st.error(f"Error executing forecast: {e}")
            return

    # --- KPI Grid ---
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Current Mandi Price</div>
                <div class="metric-value">₹ {current_price}</div>
                <div style="font-size:0.85rem; color:#6B7280;">As of {last_date.strftime('%d-%b-%y')}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
            <div class="metric-box" style="background-color: #eff6ff; border-color: #bfdbfe;">
                <div class="metric-title" style="color: #1e40af;">Predicted Next-Day</div>
                <div class="metric-value" style="color: #2563eb;">₹ {next_day_price}</div>
                <div style="font-size:0.85rem; color:#6B7280;">Target: {(last_date + pd.Timedelta(days=1)).strftime('%d-%b-%y')}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        # Determine trend style
        if pct_change > 1.5:
            trend_label = f"📈 Price Rising (+{pct_change:.1f}%)"
            trend_color = "#047857" # Green
            trend_bg = "#f0fdf4"
            trend_border = "#bbf7d0"
        elif pct_change < -1.5:
            trend_label = f"📉 Price Dropping ({pct_change:.1f}%)"
            trend_color = "#dc2626" # Red
            trend_bg = "#fef2f2"
            trend_border = "#fca5a5"
        else:
            trend_label = "➡️ Price Stable"
            trend_color = "#4b5563" # Gray
            trend_bg = "#f9fafb"
            trend_border = "#e5e7eb"
            
        st.markdown(f"""
            <div class="metric-box" style="background-color: {trend_bg}; border-color: {trend_border};">
                <div class="metric-title" style="color: {trend_color};">7-Day Price Trend</div>
                <div class="metric-value" style="color: {trend_color}; font-size:1.8rem; margin: 0.7rem 0;">{trend_label}</div>
                <div style="font-size:0.85rem; color:#6B7280;">Forecasted weekly direction</div>
            </div>
        """, unsafe_allow_html=True)

    # Smart harvest recommendation
    st.markdown("### 💡 Harvest Decision Support")
    if pct_change > 3.0:
        st.success(f"**Recommendation: HOLD HARVEST.** Prices for {selected_crop} in {selected_market} are expected to rise by **{pct_change:.1f}%** over the next week. If feasible, consider delaying sales by a few days to maximize profits.")
    elif pct_change < -3.0:
        st.error(f"**Recommendation: SELL HARVEST IMMEDIATELY.** Price forecasts show a decline of **{abs(pct_change):.1f}%** next week. Selling now will help you avoid the expected price drop.")
    else:
        st.info(f"**Recommendation: REGULAR HARVEST.** Market prices for {selected_crop} in {selected_market} are forecasted to remain stable. Proceed with your standard harvest and selling schedules.")

    # --- Historical & Forecast Chart ---
    st.markdown("### 📈 Historical Price & 7-Day Prediction Chart")
    
    # Filter last 30 historical entries for plotting
    plot_hist = market_df.tail(30).copy()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot historical actuals
    ax.plot(plot_hist["date"], plot_hist["modal_price"], label="Historical Price (Actual)", color="#10B981", marker='o', linewidth=2)
    
    # Plot forecast (connecting from the last actual price)
    forecast_dates = [last_date] + list(forecast_df["date"])
    forecast_prices = [current_price] + list(forecast_df["predicted_price"])
    ax.plot(forecast_dates, forecast_prices, label="7-Day Forecast Price", color="#3B82F6", marker='x', linestyle="--", linewidth=2)
    
    ax.set_title(f"{selected_crop} Price Forecast - {selected_market} Mandi", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (Rs./Quintal)", fontsize=11)
    ax.legend(loc="upper left")
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    st.pyplot(fig)
