import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("❤️ Heart Disease Prediction App")
st.write("Enter your clinical parameters below to assess your risk of heart disease.")

st.markdown("---")

with st.form("prediction_form"):
    st.write("### Patient Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    
    with col2:
        sex = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    
    with col3:
        cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False (0)", "True (1)"])
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", [0, 1, 2, 3])
    
    thal = st.selectbox("Thalassemia (Thal)", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])
    
    st.markdown("---")
    submitted = st.form_submit_button("Predict")

if submitted:
    sex_val = 1 if "Male" in sex else 0
    cp_val = int(cp.split("(")[1].strip(")"))
    fbs_val = 1 if "True" in fbs else 0
    exang_val = 1 if "Yes" in exang else 0
    thal_val = int(thal.split("(")[1].strip(")"))

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal_val]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write("### 🔍 Results")

    if prediction[0] == 1:
        st.error("🚨 Warning: Our model predicts you are at high risk of heart disease.")
        st.write(f"Confidence: **{prediction_proba[0][1] * 100:.2f}%**")
    else:
        st.success("✅ Good News: Our model predicts you are at low risk of heart disease.")
        st.write(f"Confidence: **{prediction_proba[0][0] * 100:.2f}%**")
