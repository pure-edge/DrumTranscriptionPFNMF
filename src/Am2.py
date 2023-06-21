from PfNmf import PfNmf
import numpy as np

def Am2(X, WD, maxIter, rh, sparsity):
    iterErr = np.zeros(maxIter)
    
    for i in range(maxIter):
        # NMF decomposition
        WD, HD, WH, HH, err = PfNmf(X, WD, [], [], [], rh, sparsity)
        
        # Keep track of error
        iterErr[i] = err[-1]
        
        # Stop criteria
        if i >= 1:
            if abs(iterErr[i - 1] - iterErr[i]) / (iterErr[0] - iterErr[i] + np.finfo(float).eps) <= 1e-3:
                iterErr[i] = 0
                break
        
        # Dictionary adaptation
        WD_new, _, WH_new, _, err = PfNmf(X, WD, HD, [], [], rh, sparsity)
        
        # Keep track of error
        iterErr[i] = err[-1]
        
        # Stop criteria
        if i >= 1:
            if abs(iterErr[i - 1] - iterErr[i]) / (iterErr[0] - iterErr[i] + np.finfo(float).eps) <= 1e-3:
                iterErr[i] = 0
                break
        
        WD = WD_new
        WH = WH_new
    
    return WD, HD, WH, HH, iterErr
