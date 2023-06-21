import numpy as np

def CrossCorrNormalized(HH, HD):
    rh, hLen = HH.shape
    rd, dLen = HD.shape
    rho = np.zeros((rd, rh))
    for i in range(rh):
        for j in range(rd):
            rho[j, i] = np.sum(HH[i, :] * HD[j, :]) / (np.linalg.norm(HH[i, :]) * np.linalg.norm(HD[j, :]))

    return rho
