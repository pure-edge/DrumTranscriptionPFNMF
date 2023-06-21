from CrossCorrNormalized import CrossCorrNormalized
import numpy as np

def TemplateAdaptation(WD, HD, WH, HH, rhoThres, adaptCoef):
    rd, numFrames = HD.shape
    rho = CrossCorrNormalized(HH, HD)
    WH_new = np.copy(WH)
    WD_new = np.copy(WD)
    for i in range(rd):
        target = np.where(rho[i, :] >= rhoThres)[0]
        if target.size > 0:
            # adaptation
            WD_new[:, i] = (1 - adaptCoef) * WD[:, i] + adaptCoef * (WH[:, target] @ rho[i, target]) / target.size

            # fill target entries in WH with random numbers
            WH_new[:, target] = np.random.rand(WH.shape[0], target.size)

    return WD_new, WH_new
