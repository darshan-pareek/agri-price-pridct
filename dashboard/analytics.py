import streamlit as st
import os

FIGURES_DIR = "reports/figures"

REPORTS_CONFIG = {
    "Price Trends Over Time (Line Chart)": {
        "file": "01_line_chart_price_trends.png",
        "description": "This line chart tracks the wholesale modal prices of Onion, Tomato, and Potato from 2020 through 2026.",
        "insights": [
            "**Tomato Price Volatility:** Tomatoes show intense annual spikes every mid-year (monsoon season). This reflects their short shelf-life and susceptibility to crop failure due to rains.",
            "**Onion Seasonal Peaks:** Onion prices show a double-peak pattern (usually autumn and early winter) reflecting the supply gap between harvest arrivals.",
            "**Potato Price Stability:** Potatoes show the lowest variance. This stability is driven by cold storage preservation, allowing traders to release stocks steadily to match demand."
        ]
    },
    "Price Frequency Distributions (Histogram)": {
        "file": "02_histogram_distributions.png",
        "description": "These histograms represent the probability distribution of modal prices for each commodity.",
        "insights": [
            "**Positive (Right) Skewness:** All three crops show highly right-skewed distributions. Most transactions occur at low price intervals, but have a long tail representing rare high-price spikes.",
            "**Implication for ML:** Standard linear regression may suffer on heavily skewed variables. Log-transformations or tree-based model ensembles (like Random Forests) are best suited to handle these right-skewed trends."
        ]
    },
    "Price Spread & Seasonality (Boxplots)": {
        "file": "03_boxplot_seasonality.png",
        "description": "These boxplots show price spreads across crops and aggregate price ranges by calendar month.",
        "insights": [
            "**Interquartile Spread:** Tomatoes have the largest box, representing high market uncertainty. Potatoes have the most compact box, reflecting low market risk.",
            "**Harvest Price Slumps:** Aggregated monthly prices dip significantly in early spring (Jan-Apr) due to winter harvest arrivals, and expand to peaks in late summer (Jul-Oct) due to supply depletion."
        ]
    },
    "Price Field Correlation Matrix (Heatmap)": {
        "file": "04_heatmap_correlation.png",
        "description": "This correlation matrix calculates the linear relationship between daily Min, Max, and Modal prices.",
        "insights": [
            "**Collinearity:** There is a near-perfect correlation (> 0.98) between reported daily minimum, maximum, and modal wholesale prices.",
            "**Feature Selection:** Because these features are highly collinear, including all three in a linear regression model would lead to numerical instability (multicollinearity). Thus, our forecasting pipeline strictly focuses on the `modal_price` column."
        ]
    },
    "Crop-wise Seasonal Patterns (Trend Chart)": {
        "file": "05_trend_chart_seasonality.png",
        "description": "This seasonal line chart isolates crop-specific averages grouped strictly by calendar month.",
        "insights": [
            "**Tomato July Spike:** Tomato prices peak in July and drop to their lowest point in March.",
            "**Onion Autumn Spike:** Onion prices climb in October/November when summer crop stocks are depleted.",
            "**Potato Storage Drawdown:** Potato prices rise slowly and steadily from March (harvest low) through November, matching the steady release of cold storage stocks."
        ]
    }
}

def render():
    st.markdown("""
        <style>
        .analytics-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #7c2d12; /* Warm brown/rust for analytics */
            margin-bottom: 1.5rem;
        }
        .insight-box {
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1.5rem;
        }
        .insight-title {
            font-weight: 700;
            color: #92400e;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="analytics-title">📊 Crop Market Analytics Reports</div>', unsafe_allow_html=True)
    st.write("Explore exploratory data analysis (EDA) reports generated from our data science processing notebooks.")

    # Dropdown to select report
    selected_report_name = st.selectbox("Select Visual Report", list(REPORTS_CONFIG.keys()))
    report = REPORTS_CONFIG[selected_report_name]
    
    image_path = os.path.join(FIGURES_DIR, report["file"])
    
    if os.path.exists(image_path):
        # Display the saved PNG from reports/figures/
        st.image(image_path, use_container_width=True)
        
        # Display report details
        st.markdown(f"**Description:** {report['description']}")
        
        # Display insights
        st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">💡 Key Diagnostic Insights</div>
                <ul>
                    {"".join(f"<li>{item}</li>" for item in report["insights"])}
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Report figure not found at `{image_path}`. Please run `src/generate_eda_plots.py` or the update pipeline to generate visualizations.")
