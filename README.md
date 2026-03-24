# ❤️ Cardio Risk Prediction (End-to-End ML Project)

 A complete machine learning project to predict cardiovascular disease risk using patient health data. This project covers the full pipeline from data preprocessing to model deployment using Streamlit.


##  Project Overview

Cardiovascular diseases are one of the leading causes of death globally. Early detection of risk factors can help in prevention and timely treatment.

This project builds a predictive model to estimate the likelihood of heart disease based on patient health metrics.


## Objectives

- Perform Exploratory Data Analysis (EDA)
- Engineer meaningful features (e.g., BMI)
- Build and compare multiple ML models
- Optimize model performance using hyperparameter tuning
- Deploy a live prediction app using Streamlit


## Dataset

- Source: Cardiovascular Disease Dataset
- Records: ~70,000 patients
- Features include:
  - Age
  - Height & Weight
  - Blood Pressure (ap_hi, ap_lo)
  - Cholesterol & Glucose levels
  - Lifestyle factors (smoking, alcohol, activity)


## Exploratory Data Analysis (EDA)

Key insights:
- Age and blood pressure are strong predictors of heart disease
- Higher cholesterol and glucose levels increase risk
- Dataset is relatively balanced
- Outliers and invalid physiological values were cleaned

---

##  Feature Engineering

- Converted age (days → years)
- Converted height (cm → meters)
- Created **BMI (Body Mass Index)**
- Combined categorical levels for better generalization
- One-hot encoding for categorical variables


## Models Used

### 1. Support Vector Machine (SVM)
- Kernel: Linear
- AUC Score: ~0.78
- Good generalization and stability

### 2. Decision Tree (Baseline)
- High training accuracy
- Overfitting observed

### 3. Tuned Decision Tree
- Hyperparameter tuning using GridSearchCV
- Reduced overfitting
- Improved test performance

## Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall (Important for healthcare)
- F1 Score
- ROC Curve (AUC ≈ 0.78)
![alt text](image.png)


##  Key Learnings

- Feature engineering (BMI) significantly improved performance
- Overfitting is common in Decision Trees and must be controlled
- Threshold tuning is critical in healthcare problems
- Recall is more important than accuracy in risk prediction

---

##  Live Demo

 https://cardioriskprediction-mithun.streamlit.app

## GitHub Repository

 https://github.com/mithun6gowda/CardioRiskPrediction


## How to Run Locally

1.git clone https://github.com/mithun6gowda/CardioRiskPrediction.git

2. pip install -r requirements.txt

3.streamlit run app.py
