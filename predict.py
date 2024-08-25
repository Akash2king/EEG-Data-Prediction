from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('_best_model.keras')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    # Get the EEG data from the request
    data = request.json
    eeg_data = np.array(data['eeg_data'])
    
    # Ensure the data is in the correct shape
    # Adjust this based on your model's input shape
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(eeg_data)
    
    # Convert prediction to emotion label
    # Adjust this based on your emotion labels
    emotion_labels = ['positive','Negative', 'Neutral']
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    
    return jsonify({
        'predicted_emotion': predicted_emotion,
        'confidence': float(np.max(prediction))
    })

if __name__ == '__main__':
    app.run(debug=True)