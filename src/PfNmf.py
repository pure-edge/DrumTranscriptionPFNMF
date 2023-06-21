from KlDivergence import KlDivergence
import numpy as np

def PfNmf(X, WD, HD, WH, HH, rh, sparsity):
    X = X + np.finfo(float).eps # make sure there's no zero frame
    numFreqX, numFrames = X.shape
    numFreqD, rd = WD.shape
    
    # initialization
    WD_update = 0
    HD_update = 0
    WH_update = 0
    HH_update = 0

    if len(WH) != 0:
        numFreqH, rh = WH.shape
    else:
        WH = np.random.rand(numFreqD, rh)
        numFreqH, _ = WH.shape
        WH_update = 1

    if numFreqD != numFreqX:
        raise ValueError('Dimensionality of WD does not match X')
    elif numFreqH != numFreqX:
        raise ValueError('Dimensionality of WH does not match X')

    if len(HD) != 0:
        WD_update = 1
    else:
        HD = np.random.rand(rd, numFrames)
        HD_update = 1

    if len(HH) != 0:
        pass
    else:
        HH = np.random.rand(rh, numFrames)
        HH_update = 1

    alpha = (rh + rd) / rd
    beta = rh / (rh + rd)

    # normalize W / H matrix
    for i in range(rd):
        WD[:, i] = WD[:, i] / np.linalg.norm(WD[:, i], 1)

    for i in range(rh):
        WH[:, i] = WH[:, i] / np.linalg.norm(WH[:, i], 1)

    count = -1
    err = np.zeros((300, 1))
    rep = np.ones((numFreqX, numFrames))

    # start iteration
    while count < 300:
        approx = alpha * WD @ HD + beta * WH @ HH

        # update
        if HD_update:
            HD = HD * ((alpha * WD).T @ (X / approx)) / ((alpha * WD).T @ rep + sparsity)
        if HH_update:
            HH = HH * ((beta * WH).T @ (X / approx)) / ((beta * WH).T @ rep)
        if WD_update:
            WD = WD * ((X / approx) @ (alpha * HD).T) / (rep @ (alpha * HD).T)
        if WH_update:
            WH = WH * ((X / approx) @ (beta * HH).T) / (rep @ (beta * HH).T)

        # normalize W matrix
        for i in range(rh):
            WH[:, i] = WH[:, i] / np.linalg.norm(WH[:, i], 1)
        for i in range(rd):
            WD[:, i] = WD[:, i] / np.linalg.norm(WD[:, i], 1)

        # calculate variation between iterations
        count = count + 1
        err[count] = KlDivergence(X, (alpha * WD @ HD + beta * WH @ HH)) + sparsity * np.linalg.norm(HD, 1)

        if count >= 1:
            if (abs(err[count] - err[count - 1]) / (err[0] - err[count] + np.finfo(float).eps)) < 0.001:
                #print(f"{abs(err[count] - err[count - 1]) / (err[0] - err[count] + np.finfo(float).eps)} < 0.001")
                break

    return WD, HD, WH, HH, err
