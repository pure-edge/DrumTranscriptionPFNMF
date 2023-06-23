from NmfDrum import NmfDrum
from FeaturesTimbre import FeaturesTimbre
import random
import csv
import os
import numpy as np


# get all 576 audio samples from strike and buzz roll folder
# get random 576 audio samples from flam and drag folder
# for each audio sample get the timbre features and append to CSV file
training_set_path = "Training_set"

file_name = 'timbre.csv'
subfolder_names = ['Strike', 'Buzz', 'Flam', 'Drag']
for index, subfolder_name in enumerate(subfolder_names):
    n = 576  # Number of random files to retrieve
    file_paths = []
    subfolder_path = os.path.join(training_set_path, subfolder_name)
    files = os.listdir(subfolder_path)
    
    if subfolder_name == 'Flam' or subfolder_name == 'Drag':
        random.shuffle(files)
        files = files[:n]

    file_paths = [os.path.join(subfolder_path, file_name) for file_name in files]

    for filePath in file_paths:
        hh, bd, sd = NmfDrum(filePath, 'Am2')
        
        print(f"number of sd onsets: {len(sd)}")
        if len(sd) > 1:
            print(f"onsets: {sd}")
        for onset in sd:
            result = FeaturesTimbre(filePath, onset)
            result = np.append(result, index+1)
            with open(file_name, 'a', newline='') as file:
                # Create a CSV writer object
                writer = csv.writer(file)
                writer.writerow(result)

