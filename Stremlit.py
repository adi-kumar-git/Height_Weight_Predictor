import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit application title
st.title("Weight Prediction Based on Height")

# Input: height in cm
height = st.number_input("Enter your height in cm", min_value=50, max_value=300, step=1)

# If a valid height is entered, make a prediction
if height > 0:
    # Convert height to inches (since the model was trained on height in inches)
    height_in_inches = height / 2.54

    # Make prediction using the trained model
    weight = model.predict([[height_in_inches]])[0]  # Predict and get the result

    # Display the predicted weight
    st.write(f"Predicted weight for a person with height {height} cm ({height_in_inches:.2f} inches) is: {weight:.2f} kg")
else:
    st.write("Please enter a valid height.")
