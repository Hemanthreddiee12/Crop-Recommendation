import streamlit as st
from joblib import load
import numpy as np

# Load the pre-trained SVM model
model = load('crop_model.joblib')

# Define the features for input
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']

# Define min and max values for each feature
feature_ranges = {
    'Nitrogen': (0, 100),
    'Phosphorus': (0, 100),
    'Potassium': (0, 100),
    'Temperature': (0, 50),
    'Humidity': (0, 100),
    'pH': (0, 14),
    'Rainfall': (0, 300)
}

# Create a Streamlit web app
def main():
    st.title("Crop Recommendation System")

    # User input for features
    user_input = {}
    for feature in features:
        min_val, max_val = feature_ranges[feature]
        # Use the same data type for min_value, max_value, and step
        value = st.slider(f"Enter {feature}:", min_value=float(min_val), max_value=float(max_val), step=0.1)
        user_input[feature] = value

    # Make prediction button
    if st.button("Get Crop Recommendation"):
        # Preprocess user input
        user_input_np = np.array(list(user_input.values())).reshape(1, -1)

        # Make prediction
        prediction = model.predict(user_input_np)

        # Display result
        st.success(f"Recommended Crop: {prediction[0]}")

# Run the app
if __name__ == '__main__':
    main()
