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
        .consumer-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a; /* Blue for consumers */
            margin-bottom: 1.5rem;
        }
        .advice-card {
            background-color: #f8fafc;
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .retail-price {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="consumer-title">🛒 Consumer Smart Budgeting Portal</div>', unsafe_allow_html=True)
    st.write("This dashboard helps households budget their monthly grocery spending by analyzing crop retail trends and comparing state-wise mandi prices.")

    if not os.path.exists(INPUT_FILE):
        st.error("⚠️ Cleaned dataset not found. Please execute the update pipeline first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    
    # Filter latest records for comparison
    latest_df = df[df["date"] == latest_date].copy()
    
    # Fallback to last available dates if latest_df is empty
    if latest_df.empty:
        latest_df = df.groupby(["Crop", "State", "District Name", "Market Name"]).last().reset_index()

    # Convert quintal price to kg price (1 Quintal = 100 kg)
    latest_df["price_per_kg"] = latest_df["modal_price"] / 100.0

    # --- 1. Current Price Summary Table ---
    st.markdown("### 📊 Latest Mandi Price Directory (per kg)")
    st.write("Prices are converted from wholesale Quintal (100 kg) to retail Kilogram (kg) equivalents for consumer convenience.")
    
    display_df = latest_df[["Crop", "State", "District Name", "Market Name", "modal_price", "price_per_kg"]].copy()
    display_df.columns = ["Crop", "State", "District", "Market", "Wholesale Price (₹/Quintal)", "Equivalent Mandi Price (₹/kg)"]
    
    # Format floats
    display_df["Wholesale Price (₹/Quintal)"] = display_df["Wholesale Price (₹/Quintal)"].astype(int)
    display_df["Equivalent Mandi Price (₹/kg)"] = display_df["Equivalent Mandi Price (₹/kg)"].map("₹ {:.2f}".format)
    
    st.dataframe(display_df, use_container_width=True)

    # --- 2. Smart Budgeting Advisor ---
    st.markdown("### 💡 Budget Advisor (7-Day Price Outlook)")
    st.write("Our ML algorithms predict weekly price directions to recommend optimal purchasing times.")
    
    advice_cols = st.columns(3)
    crops_to_advise = ["Tomato", "Onion", "Potato"]
    
    # We will compute forecast for each crop in the primary Rajasthan market for advice
    for idx, crop in enumerate(crops_to_advise):
        with advice_cols[idx]:
            # Fetch primary Rajasthan records for the crop
            crop_latest = latest_df[(latest_df["Crop"] == crop) & (latest_df["State"] == "Rajasthan")]
            if crop_latest.empty:
                crop_latest = latest_df[latest_df["Crop"] == crop]
                
            if not crop_latest.empty:
                row = crop_latest.iloc[0]
                state = row["State"]
                district = row["District Name"]
                market = row["Market Name"]
                current_p = row["modal_price"]
                
                try:
                    forecast = forecast_next_n_days(crop, state, district, market, forecast_days=7)
                    future_p = forecast.iloc[-1]["predicted_price"]
                    pct_diff = ((future_p - current_p) / current_p) * 100
                except:
                    pct_diff = 0.0
                
                # Determine recommendation
                if pct_diff > 3.0:
                    badge = "🔴 BUY NOW"
                    message = f"Prices are expected to rise by **{pct_diff:.1f}%** next week. Stock up on {crop} today to save."
                    border_color = "#dc2626"
                elif pct_diff < -3.0:
                    badge = "🟢 WAIT TO BUY"
                    message = f"Prices are expected to drop by **{abs(pct_diff):.1f}%** next week. Wait a few days to get a better price."
                    border_color = "#10b981"
                else:
                    badge = "🔵 STABLE PRICE"
                    message = f"Prices are expected to remain flat. Buy {crop} as per your standard household needs."
                    border_color = "#3b82f6"
                    
                st.markdown(f"""
                    <div class="advice-card" style="border-left-color: {border_color};">
                        <div style="font-weight:700; color:{border_color}; margin-bottom:0.5rem;">{crop.upper()} - {badge}</div>
                        <div style="font-size:0.9rem; color:#4B5563; margin-bottom:0.5rem;">{message}</div>
                        <div style="font-size:0.8rem; color:#9ca3af;">Mandi Reference: {market}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"No advice available for {crop}.")

    # --- 3. Crop Price Comparison Chart ---
    st.markdown("### 📊 Price Comparison Across States")
    selected_compare_crop = st.selectbox("Select Crop to Compare", crops_to_advise)
    
    compare_df = latest_df[latest_df["Crop"] == selected_compare_crop]
    
    if len(compare_df) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=compare_df, 
            x="Market Name", 
            y="modal_price", 
            hue="State", 
            palette="Set2", 
            ax=ax
        )
        ax.set_title(f"{selected_compare_crop} Wholesale Price Comparison across Mandis", fontweight="bold", fontsize=12)
        ax.set_ylabel("Price (₹/Quintal)")
        ax.set_xlabel("Mandi Market")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Currently, only single market records are loaded. Appending data for other states in the updates will populate this comparison chart.")
