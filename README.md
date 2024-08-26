

# EEG Emotion Detection using Deep Learning

## Overview

This project utilizes Electroencephalography (EEG) data to predict the emotional state of individuals while watching movies. The process involves preprocessing EEG signals, extracting relevant features, and applying machine learning and deep learning models to classify emotions into categories like positive, negative, and neutral.

## Features

- **EEG Data Handling**: The project handles EEG data stored in JSON format, representing brainwave signals captured during experiments.
- **Machine Learning Integration**: Employs TensorFlow to load pre-trained models for predicting emotions based on EEG data.
- **User Interface**: A Streamlit-based web application provides an interactive interface for selecting and visualizing EEG data files and predicting emotions.

## Installation

### Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- NumPy

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Akash2king/EEG-Data-Prediction.git
   cd EEG-Data-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have your EEG data files stored in a directory named `eeg_data_files`.

4. Run the Streamlit application:
   ```bash
   streamlit run streamlit_test_api.py
   ```

## Usage

1. Launch the Streamlit application by running the command above.
2. Select a JSON file containing EEG data from the dropdown menu.
3. The application will display the content of the selected file and plot the EEG signals.
4. Click the **Predict Emotion** button to classify the emotion as positive, negative, or neutral.

## Methodology

### 1. Data Collection
EEG data is collected while subjects watch various movie clips, which elicit different emotional responses. The data is stored in JSON files.

### 2. Preprocessing
The EEG data is preprocessed to filter out noise and extract relevant frequency bands (Delta, Theta, Alpha, Beta, Gamma) associated with different brain activities.

### 3. Feature Extraction
Features are extracted from the EEG signals using custom functions to represent the data meaningfully for the prediction model.

### 4. Emotion Prediction
A pre-trained deep learning model is used to predict emotions based on the extracted features. The model classifies the emotions into three categories: positive, negative, and neutral.

## Future Work

- Explore advanced deep learning architectures.
- Incorporate additional data sources for improved emotion detection.
- Enhance preprocessing techniques to better handle noise in the EEG data.

## References

- [Exploring EEG - A Beginner's Guide on Kaggle](https://www.kaggle.com/code/yorkyong/exploring-eeg-a-beginner-s-guide/notebook)

