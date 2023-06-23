from PfNmf import PfNmf
from Am1 import Am1
from Am2 import Am2
from OnsetDetection import OnsetDetection
import scipy.io

import numpy as np
import librosa

def NmfDrum(filePath, method='PfNmf', param=None):
    if param is None:
        mat_file = 'DefaultSetting.mat'  # Replace with your .mat file path
        data = scipy.io.loadmat(mat_file)
        
        param = {
            'WD': data['param'][0][0][0],
            'windowSize': 2048, # 2048
            'hopSize': 512, #512
            'lambda': [0.12, 0.12, 0.12],
            'order': [0.1, 0.1, 0.1],
            'maxIter': 20,
            'sparsity': 0,
            'rhoThreshold': 0.5,
            'rh': 50
        }
    print(f"Selected method is {method}")

    # Load audio file
    x, fs = librosa.load(filePath, sr=44100, mono=True)

    # Resample to target sample rate
    x = librosa.resample(x, orig_sr=fs, target_sr=44100)
    fs = 44100

    # Compute STFT spectrogram
    X = librosa.stft(x, n_fft=param['windowSize'], hop_length=param['hopSize'])

    # Magnitude spectrogram
    X = np.abs(X)

    if method == 'Nmf':
        param['rh'] = 0
        _, HD, _, _, _ = PfNmf(X, param['WD'], [], [], [], param['rh'], param['sparsity'])
    elif method == 'PfNmf':
        _, HD, _, _, _ = PfNmf(X, param['WD'], [], [], [], param['rh'], param['sparsity'])
    elif method == 'Am1':
        _, HD, _, _, _ = Am1(X, param['WD'], param['rh'], param['rhoThreshold'], param['maxIter'], param['sparsity'])
    elif method == 'Am2':
        _, HD, _, _, _ = Am2(X, param['WD'], param['maxIter'], param['rh'], param['sparsity'])
    #elif method == 'SaNmf':
    #    _, HD, _ = SaNmf(X, param['WD'], param['maxIter'], 4)
    #elif method == 'NmfD':
    #    PD, HD, _ = NmfD(X, param['WD'], param['maxIter'], 10)

    drumOnsetTime, drumOnsetNum = OnsetDetection(HD, fs, param['windowSize'], param['hopSize'], param['lambda'], param['order'])

    hh = drumOnsetTime[drumOnsetNum == 0]
    bd = drumOnsetTime[drumOnsetNum == 1]
    sd = drumOnsetTime[drumOnsetNum == 2]

    return hh, bd, sd
