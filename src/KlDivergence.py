import numpy as np

def KlDivergence(p, q):
    D = np.sum(p * (np.log(p + np.finfo(float).eps) - np.log(q + np.finfo(float).eps)) - p + q)
    return D