import streamlit as st
import plotly.express as px
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

clas_model= joblib.load('artifacts/classification_model.pkl')
reg_model = joblib.load('artifacts/regression_model.pkl')

def main():
    st.set_page_config(
        page_title="Placement Prediction App",
        layout="wide"
    )

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Prediction", "Dashboard"],
        index=0
    )

    if page == "Prediction":

        st.title("Student Placement and Salary Prediction")

        st.header("Input Features")

        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.number_input("SSC Percentage", 0.0, 100.0)
        hsc = st.number_input("HSC Percentage", 0.0, 100.0)
        degree = st.number_input("Degree Percentage", 0.0, 100.0)
        cgpa = st.number_input("CGPA", 0.0, 10.0)
        exam_score = st.number_input("Entrance Exam Score", 0.0, 100.0)
        tech_skill = st.number_input("Technical Skill Score", 0.0, 100.0)
        soft_skill = st.number_input("Soft Skill Score", 0.0, 100.0)
        internship = st.number_input("Internship Count", 0, 10)
        projects = st.number_input("Live Projects", 0, 20)
        experience = st.number_input("Work Experience (Months)", 0, 60)
        certifications = st.number_input("Certifications", 0, 20)
        attendance = st.number_input("Attendance Percentage", 0.0, 100.0)
        backlogs = st.number_input("Backlogs", 0, 20)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        data = {
            "gender": gender,
            "ssc_percentage": int(ssc),
            "hsc_percentage": int(hsc),
            "degree_percentage": int(degree),
            "cgpa": float(cgpa),
            "entrance_exam_score": int(exam_score),
            "technical_skill_score": int(tech_skill),
            "soft_skill_score": int(soft_skill),
            "internship_count": int(internship),
            "live_projects": int(projects),
            "work_experience_months": int(experience),
            "certifications": int(certifications),
            "attendance_percentage": float(attendance),
            "backlogs": int(backlogs),
            "extracurricular_activities": extracurricular
        }

        df = pd.DataFrame([data])

        if st.button("Make Prediction"):
            try:
                placement_pred = clas_model.predict(df)[0]
                st.subheader("Results:")

                if placement_pred == 1:
                    st.success("Accepted")

                    salary_log = reg_model.predict(df)[0]
                    salary = np.expm1(salary_log)

                    st.info(f"Salary Prediction: {salary:.2f} LPA")
                else:
                    st.error("rejected")
                    st.info("Salary Prediction: 0")

            except Exception as e:
                st.error(f"Error: {e}")

    elif page == "Dashboard":

        st.title("Dashboard")

        df = pd.read_csv("B.csv")

        st.subheader("Dataset Overview")
        st.write(df.head())

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", len(df))
        col2.metric("Placed", int(df['placement_status'].sum()))
        col3.metric("Not Placed", int((df['placement_status'] == 0).sum()))  

        st.subheader("Placement Distribution")
        st.bar_chart(df['placement_status'].value_counts())

        st.subheader("Salary Distribution (Placed Only)")
        placed_df = df[df['placement_status'] == 1]
        st.line_chart(placed_df['salary_package_lpa'])

        st.subheader("Correlation Heatmap (Numerical Features)")
        corr = df.select_dtypes(include=['int64', 'float64']).corr()
        st.dataframe(corr)        

if __name__ == "__main__":
    main()