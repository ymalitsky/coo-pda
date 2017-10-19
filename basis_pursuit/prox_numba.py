import numpy as np
from math import exp
from numba import jit, vectorize


@jit(nopython=True, nogil=True, cache=True)
def prox_l1(x, rho):
    return np.sign(x) * np.fmax(np.abs(x) - rho, 0)


@jit(nopython=True, nogil=True, cache=True)
def prox_logistic(x, rho, n_iter=3):
    """
    Proximal operators for f(x) = \sum_i ln(1+e^xi). Runs Newton's method for n_iter iterations
    """
    t = x - rho / 2.
    for i in range(n_iter):
        w = np.exp(t)
        ht = rho * w / (1. + w) + t
        dht = rho * w / (1. + w)**2 + 1
        t -= (ht - x) / dht
    return t
