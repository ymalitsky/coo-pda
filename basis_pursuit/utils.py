import numpy as np
import scipy as sp
import scipy.linalg as LA

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
