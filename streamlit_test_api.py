import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained model once when the app starts
@st.cache_resource
def load_prediction_model():
    return load_model('_best_model.keras')

model = load_prediction_model()

# Directory where JSON files are stored
directory = 'eeg_data_files'  # Replace with your directory path

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Streamlit app
st.title('EEG Data Prediction')

# Create a selection menu for the user to choose a JSON file
selected_file = st.selectbox('Select a JSON file (Example Inputs..)', json_files)

# When the user selects a file
if selected_file:
    # Construct the full file path
    file_path = os.path.join(directory, selected_file)
    
    # Read the JSON file
    with open(file_path, 'r') as json_file:
        eeg_data = json.load(json_file)
    
    # Display the selected file's content
    st.write('Selected File Content:', eeg_data)
    
    # Convert the EEG data to a NumPy array for easier manipulation and plotting
    eeg_array = np.array(eeg_data['eeg_data'])
    
    # Plot the EEG data
    st.line_chart(eeg_array)

    # Function to predict emotion
    def predict_emotion(eeg_data):
        eeg_array = np.array(eeg_data['eeg_data'])
        
        # Ensure the data is in the correct shape
        if eeg_array.ndim == 1:
            eeg_array = eeg_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(eeg_array)
        
        # Convert prediction to emotion label
        emotion_labels = ['positive', 'Negative', 'Neutral']
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        
        return predicted_emotion, float(np.max(prediction))
    
    # Send the data to the prediction function when the button is clicked
    if st.button('Predict Emotion'):
        predicted_emotion, confidence = predict_emotion(eeg_data)
        
        # Display the prediction
        if predicted_emotion == 'positive':
            st.write('Predicted Emotion: ')
            st.success(f'{predicted_emotion} ')
        elif predicted_emotion == 'Negative':
            st.write('Predicted Emotion: ')
            st.warning(f'{predicted_emotion} ')
        elif predicted_emotion == 'Neutral':
            st.write('Predicted Emotion: ')
            st.info(f'{predicted_emotion} ')
