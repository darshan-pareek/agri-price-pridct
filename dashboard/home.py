import streamlit as st
import pandas as pd
import os

INPUT_FILE = "data/processed/cleaned_master.csv"

def render():
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #4B5563;
            text-align: center;
            margin-bottom: 2rem;
        }
        .kpi-container {
            display: flex;
            justify-content: space-around;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .kpi-card {
            flex: 1;
            background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #1E3A8A;
            text-align: center;
            transition: transform 0.2s;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin: 0.5rem 0;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .crop-tomato { border-left-color: #EF4444; }
        .crop-onion { border-left-color: #8B5CF6; }
        .crop-potato { border-left-color: #F59E0B; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🌾 Smart Agricultural Price Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-Time Market Intelligence & Forecasting Platform for Indian Farmers and Consumers</div>', unsafe_allow_html=True)

    # Load data for quick KPIs
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        df["date"] = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        latest_date_str = latest_date.strftime("%d %B %Y")
        
        st.info(f"📅 Showing latest mandi market prices as of **{latest_date_str}**")
        
        # Calculate latest prices for each crop
        latest_prices = {}
        for crop in ["Tomato", "Onion", "Potato"]:
            crop_df = df[(df["Crop"] == crop) & (df["date"] == latest_date)]
            if not crop_df.empty:
                latest_prices[crop] = int(crop_df["modal_price"].mean())
            else:
                # Fallback to last available price
                last_crop_df = df[df["Crop"] == crop].sort_values("date")
                latest_prices[crop] = int(last_crop_df.iloc[-1]["modal_price"]) if not last_crop_df.empty else "N/A"
                
        # Render KPI Cards
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card crop-tomato">
                    <div class="kpi-label">🍅 Tomato Avg Price</div>
                    <div class="kpi-value">₹ {latest_prices.get('Tomato', 'N/A')}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Rs. per Quintal (100 kg)</div>
                </div>
                <div class="kpi-card crop-onion">
                    <div class="kpi-label">🧅 Onion Avg Price</div>
                    <div class="kpi-value">₹ {latest_prices.get('Onion', 'N/A')}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Rs. per Quintal (100 kg)</div>
                </div>
                <div class="kpi-card crop-potato">
                    <div class="kpi-label">🥔 Potato Avg Price</div>
                    <div class="kpi-value">₹ {latest_prices.get('Potato', 'N/A')}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Rs. per Quintal (100 kg)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Cleaned dataset not found. Please execute the data update pipeline first.")
        st.markdown("""
            <div class="kpi-container">
                <div class="kpi-card crop-tomato"><div class="kpi-label">🍅 Tomato Price</div><div class="kpi-value">₹ N/A</div></div>
                <div class="kpi-card crop-onion"><div class="kpi-label">🧅 Onion Price</div><div class="kpi-value">₹ N/A</div></div>
                <div class="kpi-card crop-potato"><div class="kpi-label">🥔 Potato Price</div><div class="kpi-value">₹ N/A</div></div>
            </div>
        """, unsafe_allow_html=True)

    # Description grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Project Mission")
        st.write("""
            Smallholder farmers in India often sell their harvests under financial duress because they lack reliable tools to anticipate market price movements. 
            Conversely, normal households struggle to budget for staple crops due to unexpected price spikes. 
            
            This system bridges the gap by transforming raw mandi price registries into actionable forecasts.
        """)
        st.markdown("### 🚀 Core Platform Features")
        st.markdown("""
            - **Farmer Dashboard**: Customized daily price checking and recursive 7-day price forecasting to optimize selling decisions.
            - **Consumer Dashboard**: Interactive state-wise price tables and buying recommendations to help households budget.
            - **Market Analytics**: Detailed historical price charts and seasonal trend maps to discover market anomalies.
        """)
        
    with col2:
        st.markdown("### 🔧 Technical Architecture")
        st.markdown("""
            This platform uses a robust, student-designed software and machine learning pipeline:
            1. **Ingestion**: Automated collector querying the Open Government Data (`data.gov.in`) API.
            2. **Preprocessing**: Missing values imputation and Z-score/IQR outlier filters.
            3. **Feature Store**: Sequential calendar extracts, grouped lag buffers, and rolling volatility metrics.
            4. **Forecasting Engine**: Autoregressive Random Forest and Gradient Boosting models tuned using grid searches with Time-Series splits.
        """)
        st.success("💡 Tip: Use the sidebar navigation to explore the Farmer and Consumer dashboards!")
