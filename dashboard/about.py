import streamlit as st

def render():
    st.markdown("""
        <style>
        .about-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .tech-badge {
            display: inline-block;
            background-color: #f3f4f6;
            color: #374151;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="about-title">ℹ️ About the Project</div>', unsafe_allow_html=True)

    st.markdown("### 🏛️ System Architecture")
    st.write("Below is the structural flow representing the data collection and model update pipeline:")
    st.code("""
+------------------+     +--------------------+     +-------------------+     +---------------------+
| Ingestion        |     | Data Preprocessing |     | Feature Store     |     | Model Training      |
|                  |     |                    |     |                   |     |                     |
| data_collector.py| --> |  data_cleaning.py  | --> |    features.py    | --> |      train.py       |
| API or Fallback  |     | IQR Outliers & NaNs|     | Time-Series Lags  |     | GridSearchCV & CV   |
| appends raw CSV  |     | standardizes schema|     | rolling averages  |     | saves models (.pkl) |
+------------------+     +--------------------+     +-------------------+     +---------------------+
                                                                                         |
                                                                                         v
                                                                              +---------------------+
                                                                              | Streamlit Dashboard |
                                                                              |                     |
                                                                              |       app.py        |
                                                                              | (Farmer & Consumer) |
                                                                              +---------------------+
    """, language="text")

    st.markdown("### 🛠️ Tech Stack & Skills Used")
    st.markdown("""
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Matplotlib / Seaborn</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Joblib</span>
        <span class="tech-badge">Python-dotenv</span>
    """, unsafe_allow_html=True)

    st.markdown("### 🗣️ Interview Readiness FAQ")
    
    with st.expander("Q: What is the main machine learning challenge in this project?"):
        st.write("""
            **Answer:** 
            The primary challenge is **avoiding data leakage** in time-series features. Because future prices are strongly correlated with past prices, if we do not shift our rolling averages and lag features by at least 1 day before making forecasts, the model will 'cheat' by reading the target value during training. 
            I resolved this by applying a `.shift(1)` to the series *before* calculating rolling windows and lag terms.
        """)
        
    with st.expander("Q: Why train separate models for each crop instead of one single model?"):
        st.write("""
            **Answer:** 
            Onion, Tomato, and Potato have fundamentally different market characteristics:
            - **Tomato** is highly perishable, subject to severe monsoon price spikes.
            - **Potato** can be preserved in cold storages for months, smoothing supply.
            - **Onion** is harvested in specific Rabi/Kharif cycles.
            
            Training a single model would dilute these unique characteristics. Separating them allows the regressors to learn the distinct coefficients and seasonal patterns of each crop.
        """)
        
    with st.expander("Q: How does the daily update script ensure reliability if the network drops?"):
        st.write("""
            **Answer:** 
            The script features a **dual-pipeline strategy**. If the data.gov.in API is unavailable or if no API key is set, the system prints a warning and falls back to a local simulation engine. 
            This engine reads the historical data, computes the last observed date, and appends realistic price fluctuations keeping seasonality intact. This ensures the dashboard always functions and retrains without crashing.
        """)
