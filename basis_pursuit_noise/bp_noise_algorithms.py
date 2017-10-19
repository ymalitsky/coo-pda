# Implementation of the primal-dual and block-coordinate primal-dual algorithms for basis pursuit problem, where the observed signal is corrupted by noise. Because of the noise, the iterates do not converge to the true solution, so we are interested in the signal error to the true solution. Analogously, due to the noise, the system Ax=b might be inconsistent, so we have to measure ||A^T(Ax-b)||.

import numpy as np
import scipy.linalg as LA
from time import process_time, time
from numba import jit, vectorize

from utils import subdif_gap, prox_l1

def pd_basis_pursuit(A, b,w, x0, sigma, tau, numb_iter=100, tol=1e-6):
    """ 
    Implementation of the primal-dual algorithm of Chambolle-Pock for the basis pursuit problem: \min |x|_1 s.t. Ax = b, where b is corrupted by noise. 

    A : 2-dimensional array
    b : measured signal (b = Aw + noise)
    w : true signal
    
    sigma: positive number, the step for the dual variable
    tau: positive number, the step for the primal variable
    
    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria includes:
    primal gap (based on the first order condition) and the
    feasibility gap ||A^T(Ax-b)||.

    In each iteration the algorithm measures the signal error to the
    true signal w and the feasibility gap.
    """
    m,n = A.shape
    x = x0
    y = np.zeros(m)

    ls_error = []
    ls_feas = []
    
    STOP = False

    for i in range(numb_iter):
        ATy = A.T.dot(y)
        x1 = prox_l1(x - tau * ATy, tau)
        z = x1 + (x1 - x)
        # Az = Ax1+
        res = A.dot(z) - b
        y += sigma * res
        x = x1

        ls_error.append(LA.norm(w-x)/LA.norm(w))
        gap1 =  subdif_gap(-ATy, x)
        ##### Change to a normal formula in the un-noise case
        gap2 = LA.norm(A.T.dot(res), ord=np.inf)
        #gap2 = LA.norm(res, ord=np.inf)
        ls_feas.append(gap2)
        #print(gap1, gap2)
        if gap1 <= tol and gap2 <= tol:
            STOP = True
            break

    if STOP:
        output = [i, gap1, gap2]
    else:
        output = [-1, gap1, gap2]

    return x, y, output, ls_error, ls_feas

# ------------------------------------------------------------------------------------
# ----------------------- Block-coordinate primal-dual algorithm ---------------------
# ------------------------------------------------------------------------------------


# block-coordinate
@jit(nopython=True, nogil=True, cache=True)
def coo_block_pd_update_numba(x, y, u, AT, n_block, dim_block, steps, sigma, ik):
    """
    Update for the block-coordinate primal-dual method for basis pursuit problem
    """

    block0 = ik * dim_block
    block1 = (ik + 1) * dim_block
    x_block = x[block0: block1].copy()

    Ai = AT[block0:block1]
    #  corresponds to the block of the size dim_block x m

    tau = steps[ik] / sigma

    block_update = prox_l1(
        x_block - tau / n_block * np.dot(Ai, y), tau / n_block)

    h = block_update - x_block
    Aih = np.dot(Ai.T, h)
    y += u + sigma * (n_block + 1) * Aih
    u += sigma * Aih
    x[block0:block1] = block_update
    return x, y, u


#@jit(nopython=True, nogil=True, cache=True)
def coo_block_pd_numba(AT, b, w, x0, steps, sigma, numb_iter=100, tol=1e-6):
    """
    Implementation of the block-coordinate primal-dual algorithm the
    basis pursuit problem: \min |x|_1 s.t. Ax = b, where b is
    corrupted by noise.

    AT : 2-dimensional array, AT = A.T.copy()
    b : measured signal (b = Aw + noise)
    w : true signal
    
    steps : array of inverse Lipschitz constants for every block A_i
    sigma: positive number, the step for the dual variable
    
    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria includes:
    primal gap (based on the first order condition) and the
    feasibility gap ||A^T(Ax-b)||.

    In each iteration the algorithm measures the signal error to the
    true signal w and the feasibility gap.

    """
    n, m = AT.shape
    x = x0.copy()
    u = sigma * (np.dot(AT.T, x0) - b)
    y = u.copy()

    ls_error = []
    ls_feas = []

    n_block = len(steps)
    dim_block = n // n_block
    STOP = False

    np.random.seed(0)
    permut = np.arange(n_block)
    for epoch in range(numb_iter):
        np.random.shuffle(permut)
        for i in range(n_block):
            ik = permut[i]
            x, y, u = coo_block_pd_update_numba(
                x, y, u, AT, n_block, dim_block, steps, sigma, ik)
            
        #f_gap = 1 / sigma * np.sqrt(np.dot(u, u))
        tmp1 = AT.dot(u)
        f_gap = 1 / sigma *  LA.norm(tmp1, ord=np.inf)
        ls_error.append(LA.norm(w-x)/LA.norm(w))
        ls_feas.append(f_gap)
        #f_gap = 1 / sigma * np.sqrt(np.dot(tmp1, tmp1))
        #f_gap = 1 / sigma * LA.norm(u, ord=np.inf)
        #print(f_gap)
        # we don't want to compute s_gap in every iteration, since it
        # requires computing A.T.dot(y). We compute it only if the
        # feasibility gap is already small.
        if f_gap <= tol:
            s_gap = subdif_gap(-np.dot(AT, y), x)
            #s_gap = 0 
            if s_gap <= tol:
                STOP = True
                break

    if STOP:
        # n_epoch = i // n_block
        output = [epoch, s_gap, f_gap]
    else:
        f_gap = 1 / sigma * LA.norm(u, ord=np.inf)
        s_gap = subdif_gap(-np.dot(AT, y), x)
        # means that the algorithm does not converge within N*n_batch
        # iterations
        epoch = -1
        output = [epoch, s_gap, f_gap]

    return x, y, output, ls_error, ls_feas
