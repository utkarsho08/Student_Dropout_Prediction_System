import streamlit as st
import json
import pandas as pd
from config.settings import METRICS_PATH, ASSETS_DIR
from utils.theme import apply_custom_theme
from utils.ui import render_header

st.set_page_config(page_title="Model Evaluation", page_icon="", layout="wide")
apply_custom_theme()
render_header("Model Evaluation & Benchmarks", "Performance metrics across trained classifiers")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    best_model = metrics.pop("Best_Model", None)
    df_metrics = pd.DataFrame(metrics).T
    
    st.subheader(f"Best Performing Classifier: **{best_model}**")
    st.dataframe(df_metrics.style.highlight_max(axis=0, color="#1e3a8a"), use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve Comparison")
    if (ASSETS_DIR / "roc_curve.png").exists():
        st.image(str(ASSETS_DIR / "roc_curve.png"), use_container_width=True)

with col2:
    st.subheader("Confusion Matrix (Best Model)")
    if (ASSETS_DIR / "confusion_matrix.png").exists():
        st.image(str(ASSETS_DIR / "confusion_matrix.png"), use_container_width=True)

st.markdown("---")
st.subheader("Key Drivers of Student Attrition (Feature Importance)")
if (ASSETS_DIR / "feature_importance.png").exists():
    st.image(str(ASSETS_DIR / "feature_importance.png"), use_container_width=True)