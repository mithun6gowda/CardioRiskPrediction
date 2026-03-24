import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Cardio Risk Prediction")

# Inputs
age = st.number_input("Age")
height = st.number_input("Height (in meters)")
weight = st.number_input("Weight (in kg)")
ap_hi = st.number_input("Systolic BP")
ap_lo = st.number_input("Diastolic BP")

gender = st.selectbox("Gender (1=Male, 2=Female)", [1, 2])
cholesterol = st.selectbox("Cholesterol (1=Normal, 2=Above, 3=High)", [1, 2, 3])
gluc = st.selectbox("Glucose (1=Normal, 2=Above, 3=High)", [1, 2, 3])
smoke = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
alco = st.selectbox("Alcohol (0=No, 1=Yes)", [0, 1])
active = st.selectbox("Physical Activity (0=No, 1=Yes)", [0, 1])

if st.button("Predict"):

    # Create input dictionary
    data = {
        'age': age,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,

        # Gender dummy
        'gender_1': 1 if gender == 1 else 0,

        # Cholesterol dummies
        'cholesterol_1': 1 if cholesterol == 1 else 0,
        'cholesterol_2': 1 if cholesterol == 2 else 0,

        # Glucose dummies
        'gluc_1': 1 if gluc == 1 else 0,
        'gluc_2': 1 if gluc == 2 else 0,

        # Binary
        'smoke_1': smoke,
        'alco_1': alco,
        'active_1': active
    }

    # Convert to DataFrame
    data_df = pd.DataFrame([data])

    # Ensure correct column order
    data_df = data_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prob = model.predict_proba(data_df)[0][1]
    st.write("Probability:", prob)
    st.subheader(f"Risk Probability: {prob:.2f}")
    st.success("High Risk" if prob > 0.5 else "Low Risk")