"""Dashboard CSS injected once at startup."""
import streamlit as st

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FF9800; font-weight: bold; }
    .status-bad { color: #F44336; font-weight: bold; }

    /* Mobile Responsive Styles */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* Smaller metrics on mobile */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
        }
        /* Tighter padding */
        .block-container {
            padding: 1rem !important;
        }
        /* Smaller expander headers */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
        }
    }

    /* Tablet responsiveness */
    @media (max-width: 1024px) and (min-width: 769px) {
        .main-header {
            font-size: 2rem;
        }
        /* 2 columns on tablet */
        [data-testid="column"] {
            min-width: 45% !important;
        }
    }

    /* Risk Score styling */
    .risk-score-gauge {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(145deg, #f5f5f5, #e0e0e0);
        margin-bottom: 10px;
    }

    /* Alert styling */
    .alert-badge {
        padding: 5px 10px;
        border-radius: 5px;
        margin: 3px 0;
        font-size: 0.85rem;
    }
</style>
"""


def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
