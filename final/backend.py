from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained models for prolongation, repetition, and block
prolongation_model = load_model('prolongation_model.h5')
repetition_model = load_model('repetition_model.h5')
block_model = load_model('block_model.h5')

# Function to preprocess audio
def preprocess_audio(audio_file):
    signal, sr = librosa.load(audio_file, sr=16000)  # Resample to 16kHz
    return signal, sr

# Function to extract features from audio
def extract_features(signal, sr):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    # Extract pitch
    pitch = librosa.yin(signal, fmin=75, fmax=600)
    mean_pitch = np.mean(pitch)
    # Extract intensity
    intensity = np.sum(np.abs(signal))
    mean_intensity = intensity / len(signal)
    # Calculate speech rate
    speech_rate = np.count_nonzero(signal) / len(signal)  # Simplified speech rate calculation
    # Flatten MFCCs to a 1D array
    mfccs_flattened = mfccs.T.flatten()
    return np.concatenate(([mean_pitch, mean_intensity, speech_rate], mfccs_flattened))

# Function to generate report based on severity predictions
def generate_report(prolongation_severity, repetition_severity, block_severity):
    # Calculate average severity
    combined_severity = (prolongation_severity + repetition_severity + block_severity) / 3.0
    # Determine severity label
    if combined_severity < 0.33:
        label = 'Low'
        description = 'The severity of stuttering is low.'
    elif combined_severity < 0.67:
        label = 'Moderate'
        description = 'The severity of stuttering is moderate.'
    else:
        label = 'Severe'
        description = 'The severity of stuttering is severe.'
    return {'severity': combined_severity, 'label': label, 'description': description}

# Route for uploading video and getting severity report
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        audio_path = os.path.join('uploads', f.filename)
        f.save(audio_path)
        
        # Preprocess audio
        signal, sr = preprocess_audio(audio_path)
        # Extract features
        features = extract_features(signal, sr)
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        # Reshape features for CNN layer
        features_reshaped = features_scaled.reshape(1, features_scaled.shape[0], 1)
        
        # Predict severity for each model
        prolongation_severity = prolongation_model.predict(features_reshaped)[0][0]
        repetition_severity = repetition_model.predict(features_reshaped)[0][0]
        block_severity = block_model.predict(features_reshaped)[0][0]
        
        # Generate combined report
        report = generate_report(prolongation_severity, repetition_severity, block_severity)
        
        # Return severity report
        return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)
