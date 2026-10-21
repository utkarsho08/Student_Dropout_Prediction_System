import streamlit as st
from utils.theme import apply_custom_theme
from utils.ui import render_header, render_section_title

st.set_page_config(page_title="System Documentation", layout="wide")
apply_custom_theme()

render_header(
    title="System Architecture & Methodology Specification",
    subtitle="Technical reference for data pipelines, feature engineering, and model governance.",
    category="Module 04 // Documentation"
)

col1, col2 = st.columns(2)

with col1:
    render_section_title("1. Problem Formulation")
    st.write("""
    Student attrition in higher education impacts institutional graduation rates and student career trajectories. 
    This system formulates retention monitoring as a supervised binary classification task, estimating the probability 
    of student dropout prior to the third semester using early academic and socio-economic telemetry.
    """)
    
    render_section_title("2. Feature Engineering Schema")
    st.markdown("""
    - **Academic Indicators**: `CGPA_1st_Sem`, `CGPA_2nd_Sem`, `Active_Backlogs`
    - **Engagement Telemetry**: `Attendance_Rate`, `Counseling_Attended`, `Hostel_Resident`
    - **Socio-Economic Factors**: `Tuition_Fees_Up_To_Date`, `Scholarship_Holder`, `Family_Income_LPA`
    - **Demographic Controls**: `Age_at_Enrollment`, `Gender`
    """)

with col2:
    render_section_title("3. Machine Learning Pipeline")
    st.markdown("""
    1. **Standardization**: Continuous variables are zero-centered and scaled to unit variance via `StandardScaler` persisted at `models/scaler.pkl`.
    2. **Categorical Encoding**: Binary indicators are deterministically mapped to integer representations.
    3. **Model Ensemble Suite**: Trains **Logistic Regression** (baseline linear boundary), **Random Forest** (bagged decision trees), and **XGBoost** (gradient-boosted trees).
    4. **Automated Selection**: Classifiers are evaluated on a stratified 20% holdout split; the model maximizing F1-Score is promoted to `models/best_model.pkl`.
    """)
