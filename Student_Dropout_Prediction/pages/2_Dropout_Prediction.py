import streamlit as st
from utils.theme import apply_custom_theme
from utils.ui import render_header
from utils.predictor import predict_dropout, MODEL_MAP

st.set_page_config(page_title="Dropout Prediction", page_icon="", layout="wide")
apply_custom_theme()
render_header("Real-Time Student Dropout Risk Predictor", "Assess individual student vulnerability and generate intervention recommendations")

model_choice = st.sidebar.selectbox("Select Prediction Model", list(MODEL_MAP.keys()))

with st.form("student_prediction_form"):
    st.subheader("Academic & Demographic Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age at Enrollment", min_value=16, max_value=35, value=19)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cgpa_1 = st.slider("1st Semester CGPA", 0.0, 10.0, 7.2, 0.1)
        cgpa_2 = st.slider("2nd Semester CGPA", 0.0, 10.0, 6.8, 0.1)
        
    with col2:
        attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, 0.5)
        backlogs = st.number_input("Active Backlogs", min_value=0, max_value=15, value=0)
        counseling = st.selectbox("Counseling Sessions Attended?", ["Yes", "No"])
        
    with col3:
        income = st.number_input("Family Annual Income (LPA)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
        tuition_up_to_date = st.selectbox("Tuition Fees Up To Date?", ["Yes", "No"])
        scholarship = st.selectbox("Scholarship Holder?", ["Yes", "No"])
        hostel = st.selectbox("Hostel Resident?", ["Yes", "No"])
        
    submitted = st.form_submit_button("Analyze Dropout Risk", use_container_width=True)

if submitted:
    input_data = {
        "Age_at_Enrollment": age,
        "CGPA_1st_Sem": cgpa_1,
        "CGPA_2nd_Sem": cgpa_2,
        "Attendance_Rate": attendance,
        "Active_Backlogs": backlogs,
        "Family_Income_LPA": income,
        "Gender": gender,
        "Scholarship_Holder": scholarship,
        "Tuition_Fees_Up_To_Date": tuition_up_to_date,
        "Hostel_Resident": hostel,
        "Counseling_Attended": counseling
    }
    
    pred, prob = predict_dropout(input_data, model_choice)
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if pred == 1:
            st.error(f"High Dropout Risk Detected\n**Predicted Probability:** `{prob * 100:.1f}%`")
        else:
            st.success(f"Student Likely to be Retained\n**Dropout Risk Probability:** `{prob * 100:.1f}%`")
        st.progress(min(max(prob, 0.0), 1.0))
        
    with res_col2:
        st.subheader("Recommended Faculty Interventions")
        recommendations = []
        if attendance < 75.0:
            recommendations.append("**Attendance Alert**: Schedule mandatory academic advisor check-in for low attendance.")
        if backlogs >= 2:
            recommendations.append("**Academic Support**: Enroll student in peer-tutoring and backlog remediation workshops.")
        if tuition_up_to_date == "No":
            recommendations.append("**Financial Aid**: Connect student with the financial aid office for installment plans or emergency grants.")
        if cgpa_2 < cgpa_1 - 0.5:
            recommendations.append("**Performance Drop**: Sharp CGPA decline between semesters; recommend faculty mentorship.")
        if not recommendations:
            recommendations.append("Student metrics are healthy. Continue standard academic monitoring.")
            
        for rec in recommendations:
            st.write(rec)