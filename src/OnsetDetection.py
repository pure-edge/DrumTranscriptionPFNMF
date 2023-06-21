from MedianThres import MedianThres
from PeakPicking import PeakPicking

import numpy as np

def OnsetDetection(HD, fs, windowSize, hopSize, _lambda, order):
    numDrum, numFrames = HD.shape
    myTmpResults = []
    myTmpTrans = []
    order = np.floor_divide(order, hopSize/fs).astype(int) # sec to blocks
    for i in range(numDrum):
        nvt = HD[i, :]
        order_current = order[i]
        lambda_current = _lambda[i]

        # adaptive thresholding
        threshold = MedianThres(nvt, order_current, lambda_current)

        # peak picking
        onsetTimeInSec, _ = PeakPicking(nvt, threshold, fs, windowSize, hopSize)

        numOnsets = onsetTimeInSec.shape[0]
        myTmpTrans = np.full(numOnsets, i)
        myTmpResults.append(np.vstack((onsetTimeInSec, myTmpTrans)))

    myTmpResults = np.concatenate(myTmpResults, axis=1)
    tmp = myTmpResults[:, myTmpResults[0, :].argsort()]
    myTmpResults = tmp

    drumOnsetTime = myTmpResults[0, :]
    drumOnsetNum = myTmpResults[1, :]

    return drumOnsetTime, drumOnsetNum
