import pickle
from NmfDrum import NmfDrum
from FeaturesTimbre import FeaturesTimbre
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = None
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# read file
filePathAudio = '122_minus-one_celtic-rock_sticks.wav'

# Annotation-Informed Testing
file_path_annotation = '122_min.txt'  # Replace with the path to your text file

with open(file_path_annotation, 'r') as file:
    for line in file:
        line = line.strip()
        split_line = line.split()
        
        onset = float(split_line[0])
        features = FeaturesTimbre(filePathAudio, onset)
        features = np.array(features).reshape(1, -1)
        prediction = loaded_model.predict(features)
        
        labels = ['Strike', 'Buzz Roll', 'Flam', 'Drag']
        prediction_text = labels[prediction[0]-1]
        print(f"actual: {split_line[3]}, predicted: {prediction_text}")


### Real-World Testing
# transcriptions
hh, bd, sd = NmfDrum(filePathAudio, 'Am2')

for onset in sd:
    features = FeaturesTimbre(filePathAudio, onset)
    features = np.array(features).reshape(1, -1)
    prediction = loaded_model.predict(features)
    
    labels = ['Strike', 'Buzz Roll', 'Flam', 'Drag']
    prediction_text = labels[prediction[0]-1]
    print(f"onset: {onset}, predicted: {prediction_text}")
    
