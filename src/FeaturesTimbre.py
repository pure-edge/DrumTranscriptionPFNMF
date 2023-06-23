import numpy as np
import librosa

def FeaturesTimbre(filePath, onset):
    windowSize = 512 # 2048
    hopSize = 128 #512

    # Load audio file
    signal, sr = librosa.load(filePath, sr=44100, mono=True)

    # get a 400-millisecond segment from the audio signal
    start = onset - 0.2 if onset - 0.2 >= 0 else onset
    start_sample = int((start) * sr)
    end = onset + 0.2 if onset - 0.2 >= 0 else onset + 0.4
    end_sample = start_sample + int(end * sr)
    segment = signal[start_sample:end_sample]
    
    # re-position to the onset towards the highest value in the segment
    max_value = np.max(segment)
    max_index = np.where(segment == max_value)[0]
    max_index = max_index + start_sample # indexes adjust to match signal index
    # Find the index of the value closest to max_value
    index_closest = np.argmin(np.abs(max_index - start_sample))
    # Get the closest value
    max_index = max_index[index_closest]
    onset = librosa.samples_to_time(max_index, sr=sr)
    
    # adjust the segment with the new onset
    start = onset - 0.2 if onset - 0.2 >= 0 else onset
    start_sample = int((start) * sr)
    end = onset + 0.2 if onset - 0.2 >= 0 else onset + 0.4
    end_sample = start_sample + int(end * sr)
    segment = signal[start_sample:end_sample]
    
    # Compute the spectrogram of the audio segment
    
    spectrogram = librosa.stft(segment, n_fft=windowSize, hop_length=hopSize)
    spectrogram = np.abs(spectrogram)

    centroid = librosa.feature.spectral_centroid(S=spectrogram)[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrogram)[0]
    flux = librosa.onset.onset_strength(S=spectrogram)
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

    mean_values = np.mean(mfcc, axis=1)
    std_values = np.std(mfcc, axis=1)
    mfccs = np.concatenate((mean_values, std_values))
    
    result = [np.mean(centroid), np.std(centroid), 
            np.mean(rolloff), np.std(rolloff),
            np.mean(flux), np.std(flux),
            np.mean(zcr), np.std(zcr),
            ]
    for val in mfccs:
        result.append(val)
        
    return result
