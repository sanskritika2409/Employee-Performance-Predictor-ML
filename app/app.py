import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("../models/model.pkl")

st.title("💼 Employee Performance Predictor")

st.write("Enter employee details:")

# Inputs
age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience (years)", 1, 20, 5)
department = st.selectbox("Department", ["HR", "IT", "Sales"])
salary = st.number_input("Salary", 30000, 120000, 50000)
training = st.slider("Training Hours", 0, 100, 40)
projects = st.slider("Projects", 1, 10, 5)

# Encode department manually
dept_map = {"HR": 0, "IT": 1, "Sales": 2}
dept_val = dept_map[department]

# Prediction
if st.button("Predict Performance"):
    input_data = np.array([[age, experience, dept_val, salary, training, projects]])
    prediction = model.predict(input_data)

    labels = {0: "High", 1: "Low", 2: "Medium"}

    st.success(f"Predicted Performance: {labels[prediction[0]]}")