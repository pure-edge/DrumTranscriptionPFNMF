from TemplateAdaptation import TemplateAdaptation
from PfNmf import PfNmf
import numpy as np

def Am1(X, WD, rh, rhoThres, maxIter, sparsity):
    iterErr = np.zeros(maxIter)
    
    for i in range(maxIter):
        # NMF decomposition
        WD, HD, WH, HH, err = PfNmf(X, WD, [], [], [], rh, sparsity)
        
        # Keep track of error
        iterErr[i] = err[-1]
        
        # Stop criteria
        if i >= 1:
            if abs(iterErr[i - 1] - iterErr[i]) / (iterErr[0] - iterErr[i] + np.finfo(float).eps) <= 10e-3:
                iterErr[i] = 0
                break
        
        # Dictionary extraction
        adaptCoef = 1 / (2 ** (i+1))  # Start from 0.2
        WD_new, WH_new = TemplateAdaptation(WD, HD, WH, HH, rhoThres, adaptCoef)
        
        # NMF decomposition
        _, _, _, _, err = PfNmf(X, WD_new, [], WH_new, [], rh, sparsity)
        
        # Keep track of error
        iterErr[i] = err[-1]
        
        # Stop criteria
        if i >= 1:
            if abs(iterErr[i - 1] - iterErr[i]) / (iterErr[0] - iterErr[i] + np.finfo(float).eps) <= 10e-3:
                iterErr[i] = 0
                break
        
        # Adaptation
        WD = WD_new
        WH = WH_new
    
    return WD, HD, WH, HH, iterErr


