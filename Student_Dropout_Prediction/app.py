import streamlit as st
import pandas as pd
import json
from config.settings import APP_TITLE, APP_ICON, DATA_PATH, METRICS_PATH
from utils.theme import apply_custom_theme
from utils.ui import render_header
from training.train_models import train_and_evaluate

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
apply_custom_theme()

if not DATA_PATH.exists() or not METRICS_PATH.exists():
    with st.spinner("Initializing dataset and training machine learning models..."):
        train_and_evaluate()

render_header(
    "Student College Dropout Prediction System",
    "AI-Powered Early Warning & Academic Retention Analytics Dashboard"
)

df = pd.read_csv(DATA_PATH)
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

best_model_name = metrics.get("Best_Model", "Random Forest")
best_acc = metrics[best_model_name]["Accuracy"] * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students Analyzed", f"{len(df):,}")
col2.metric("Overall Dropout Rate", f"{(df['Dropout'].mean() * 100):.1f}%")
col3.metric("Best Model", best_model_name)
col4.metric("Best Model Accuracy", f"{best_acc:.1f}%")

st.markdown("---")
st.subheader("Quick Navigation Guide")
st.markdown("""
-  1_EDA_Dashboard      : Explore interactive distributions of attendance, CGPA, backlogs, and financial factors.
-  2_Dropout_Prediction : Input live student parameters to evaluate dropout risk and get actionable intervention advice.
-  3_Model_Evaluation   : Compare Logistic Regression, Random Forest, and XGBoost across ROC curves, F1 scores, and feature importances.
-  4_About_Project      : Documentation on methodology, system architecture, and retention strategies.
""")