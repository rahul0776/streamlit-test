import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained SVR model and scaler
with open('./app/svr_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the column order used during training
TRAINED_COLUMNS = [
    'Age',
    'Education_Bachelor’s degree',
    'Education_Master’s degree',
    'Education_Doctoral degree',
    'ExperienceLevel_Entry level',
    'ExperienceLevel_Mid-Senior level',
    'ExperienceLevel_Executive'
]

def app():
    st.set_page_config(page_title="Job Level Predictor", layout="wide")

    st.sidebar.header("Input Features")
    st.sidebar.markdown("Enter the features below:")

    # User Inputs
    age = st.sidebar.slider("Age", 18, 75, 30)
    education_level = st.sidebar.selectbox(
        "Education Level", [
            "Bachelor’s degree", 
            "Master’s degree", 
            "Doctoral degree"
        ]
    )
    experience_level = st.sidebar.selectbox(
        "Experience Level", [
            "Entry level", 
            "Mid-Senior level", 
            "Executive"
        ]
    )

    # One-hot encode the categorical inputs
    input_data = pd.DataFrame(columns=TRAINED_COLUMNS)
    input_data.loc[0] = 0  # Initialize all columns to 0

    # Set the values for the user input
    input_data['Age'] = age
    input_data[f'Education_{education_level}'] = 1
    input_data[f'ExperienceLevel_{experience_level}'] = 1

    # Display input data
    st.write("### Input Data")
    st.write(input_data)

    # Scale the input data
    try:
        # Pass the raw numpy array to avoid feature name issues
        scaled_data = scaler.transform(input_data.values)
        st.write("### Scaled Data")
        st.write(scaled_data)

        # Predict button
        if st.button("Predict Job Level"):
            prediction = model.predict(scaled_data)
            st.write(f"### Predicted Job Level: {round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Error scaling input data: {e}")

if __name__ == "__main__":
    app()
