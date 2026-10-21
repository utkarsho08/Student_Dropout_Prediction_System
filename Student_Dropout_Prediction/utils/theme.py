import streamlit as st

def apply_custom_theme():
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #1e293b;
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            color: #f8fafc;
        }
        </style>
    """, unsafe_allow_html=True)