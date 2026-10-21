import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import DATA_PATH, ASSETS_DIR
from utils.theme import apply_custom_theme
from utils.ui import render_header

st.set_page_config(page_title="EDA Dashboard", page_icon="", layout="wide")
apply_custom_theme()
render_header(" Exploratory Data Analysis", "Uncover patterns driving student attrition and academic retention")

df = pd.read_csv(DATA_PATH)
df["Status"] = df["Dropout"].map({0: "Retained", 1: "Dropout"})

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df, x="Attendance_Rate", color="Status", barmode="overlay",
        title="Attendance Rate Distribution by Student Status",
        color_discrete_map={"Retained": "#10b981", "Dropout": "#ef4444"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(
        df, x="Status", y="CGPA_2nd_Sem", color="Status",
        title="2nd Semester CGPA vs Dropout Status",
        color_discrete_map={"Retained": "#10b981", "Dropout": "#ef4444"}
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(
        df, x="Active_Backlogs", color="Status", barmode="group",
        title="Impact of Active Backlogs on Student Dropout",
        color_discrete_map={"Retained": "#10b981", "Dropout": "#ef4444"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.histogram(
        df, x="Tuition_Fees_Up_To_Date", color="Status", barmode="group",
        title="Tuition Fee Payment Status vs Attrition",
        color_discrete_map={"Retained": "#10b981", "Dropout": "#ef4444"}
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.subheader("Feature Correlation Heatmap")
if (ASSETS_DIR / "heatmap.png").exists():
    st.image(str(ASSETS_DIR / "heatmap.png"), use_container_width=True)