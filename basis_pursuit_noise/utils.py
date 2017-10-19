import numpy as np
import scipy as sp
import scipy.linalg as LA
from math import exp

from numba import jit, vectorize


@vectorize()
def subdif_gap_1d(w, x):
    """
    Horrible code, but otherwise does not work with numba
    """
    eps = 1e-10
    if x < 0 - eps:
        d = w + 1
    elif x > 0 + eps:
        d = w - 1
    else:
        if -1 <= w <= 1:
            d = 0
        elif w > 1:
            d = w - 1
        else:
            d = -w - 1

    return d


@jit(nopython=True, nogil=True, cache=True)
def subdif_gap(w, x):
    "Returns infinity distance between a vector and a subdifferential"
    dist = subdif_gap_1d(w, x)
    return np.max(dist)


def subdif_gap2(w, x, eps=1e-10):
    """
    Computes the distance between w and the subdifferential of l1-norm at x.
    """

    ind1 = x < 0 - eps
    ind2 = x > 0 + eps

    ind3 = 1 - (ind1 + ind2)

    d1 = max(np.abs(w[ind1] + 1)) if np.size(w[ind1]) != 0 else 0
    d2 = max(np.abs(w[ind2] - 1)) if np.size(w[ind2]) != 0 else 0

    d3 = max(w[ind3] - np.clip(w[ind3], -1, 1))
    return max(d1, d2, d3)


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


def proj_ball2(x, r):
    """
    Compute projection of x onto the closed ball B(0, r)
    """
    dist = LA.norm(x)
    if dist <= r:
        return x
    else:
        return r * x / dist 


def proj_ball(x, r):
    """
    Compute projection of x onto the closed ball B(0, r)
    """
    return np.clip(x,-r, r)

def haar_matrix(size):
    level = int(np.ceil(np.log2(size)))
    H = np.array([1.])[:, None]
    NC = 1. / np.sqrt(2.)
    LP = np.array([1., 1.])[:, None] 
    HP = np.array([1., -1.])[:, None]
    for i in range(level):
        H = NC * np.hstack((np.kron(H, LP), np.kron(np.eye(len(H)),HP)))
    H = H.T
    return H
