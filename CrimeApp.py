# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Gradient Boosting model
best_gb_model = joblib.load("best_gb_model.h5")

# Define a function for prediction
def predict_crime_category(features):
    # Assuming features is a dictionary containing input features

    # Example: Preprocess the input features
    # You need to adjust this based on your actual preprocessing steps during training
    input_data = pd.DataFrame([features])

    # Example: Handle categorical encoding (assuming you used one-hot encoding during training)
    input_data_encoded = encoder.transform(input_data[['LOCATION_TYPE']])

    # Make predictions
    prediction = best_gb_model.predict(input_data_encoded)

    return prediction[0]


# Streamlit UI
def main():
    st.title("Crime Prediction App")

    # Get user input for crime features
    user_input = {}

    # Example: Get user input using Streamlit text_input
    user_input['LOCATION_TYPE'] = st.text_input("Enter LOCATION_TYPE")

    # Add more input fields based on your feature set

    # Make a prediction when the user clicks the button
    if st.button("Predict"):
        prediction = predict_crime_category(user_input)
        st.write(f"Predicted Crime Category: {prediction}")

if __name__ == "__main__":
    main()
