import streamlit as st

def render_header(title: str, subtitle: str, category: str = "Institutional Intelligence Suite"):
    st.markdown(f"""
        <div class="page-header">
            <div class="category-tag">{category}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def render_kpi(label: str, value: str, subtitle: str = ""):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

def render_section_title(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
