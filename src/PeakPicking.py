import numpy as np
from scipy.signal import find_peaks, peak_prominences

def PeakPicking(nvt, threshold, fs, windowSize, hopSize):
    if nvt.ndim == 1:
        nvt = np.reshape(nvt, (1, -1))
    # Find peaks row by row
    numBlocks = nvt.shape[1]
    
    # Initialization
    hopTime = hopSize / fs  # Hop time
    winTime = windowSize / fs  # Window time
    t = np.arange(0, numBlocks + 1) * hopTime
    
    tmp = np.copy(nvt)
    tmp[tmp <= threshold] = 0
    onsetTimeInFrame, _ = find_peaks(tmp.flatten())
    onsetTimeInSec = t[onsetTimeInFrame]
    
    return onsetTimeInSec, onsetTimeInFrame
