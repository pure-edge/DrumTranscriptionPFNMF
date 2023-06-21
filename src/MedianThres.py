import numpy as np

def MedianThres(nvt, order, lambda_val):
    if nvt.ndim == 1:
        nvt = np.reshape(nvt, (1, -1))
    
    m, numPoints = nvt.shape
    threshold = np.zeros((m, numPoints))
    maxVal = np.max(nvt, axis=1)
    
    for i in range(numPoints):
        med = np.median(nvt[:, max(0, i - order + 1):i+1], axis=1)
        threshold[:, i] = lambda_val * maxVal + med
        pass
    
    # compensate the delay of the threshold
    shiftSize = round(0.5 * order)  # 1/2 order size
    threshold[:, :numPoints - shiftSize] = threshold[:, shiftSize:]
    
    return threshold
