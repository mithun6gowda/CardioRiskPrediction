import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Cardio Risk Prediction App")

age = st.number_input("Age")
height = st.number_input("Height (m)")
weight = st.number_input("Weight (kg)")
ap_hi = st.number_input("Systolic BP")
ap_lo = st.number_input("Diastolic BP")
cholesterol = st.selectbox("Cholesterol", [1, 2])
gluc = st.selectbox("Glucose", [1, 2])
smoke = st.selectbox("Smoking", [0, 1])
alco = st.selectbox("Alcohol", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

if st.button("Predict"):
    bmi = weight / (height ** 2)

    data = np.array([[age, height, ap_hi, ap_lo, bmi,
                      cholesterol, gluc, smoke, alco, active]])

    prob = model.predict_proba(data)[0][1]

    st.subheader(f"Risk Probability: {prob:.2f}")
    st.success("High Risk" if prob > 0.5 else "Low Risk")