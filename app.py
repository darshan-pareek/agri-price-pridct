import streamlit as st
from dashboard import home, farmer, consumer, analytics, about

# Setup Streamlit page configuration
st.set_page_config(
    page_title="Smart Agricultural Price Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar branding header
st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="color: #1E3A8A; margin-bottom: 0;">🌾 AgriPredict</h2>
        <small style="color: #6B7280;">Market Intelligence v1.0</small>
    </div>
""", unsafe_allow_html=True)

# Navigation radio choices
page = st.sidebar.radio(
    "Navigation Menu",
    ["Home", "Farmer Dashboard", "Consumer Dashboard", "Crop Analytics", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Run the `daily_update.py` script in your terminal to fetch the latest mandi data and retrain models automatically.")

# Route to the selected page module
if page == "Home":
    home.render()
elif page == "Farmer Dashboard":
    farmer.render()
elif page == "Consumer Dashboard":
    consumer.render()
elif page == "Crop Analytics":
    analytics.render()
elif page == "About Project":
    about.render()
