# This module contains implementation of the primal-dual algorithm and itc coordinate extensions for the basis pursuit problem.

import numpy as np
import scipy.linalg as LA

from time import process_time, time
from numba import jit, vectorize
from prox_numba import prox_l1
from utils import subdif_gap


def pd_basis_pursuit(A, b, x0, sigma, tau, numb_iter=100, tol=1e-6):
    """
    Implementation of the primal-dual algorithm of Chambolle and Pock for basis pursuit problem:
    \min |x|_1 s.t. Ax = b
    A : 2-dimensional array

    sigma: positive number, the step for the dual variable
    tau: positive number, the step for the primal variable
    
    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria includes:
    primal gap (based on the first order condition) and the
    feasibility gap ||Ax-b||.
    """
    m,n = A.shape
    x = x0
    #y = A.dot(x0) - b
    y = np.zeros(m)

    STOP = False

    for i in range(numb_iter):
        ATy = A.T.dot(y)
        x1 = prox_l1(x - tau * ATy, tau)
        z = x1 + (x1 - x)
        # Az = Ax1+
        res = A.dot(z) - b
        y += sigma * res
        x = x1

        # compute the distance between subdifferential and a current point
        gap1 = subdif_gap(-ATy, x)
        ### Change to a normal formula in the un-noise case
        #gap2 = LA.norm(A.T.dot(res))
        gap2 = LA.norm(res, ord=np.inf)
        #print(gap1, gap2)
        if gap1 <= tol and gap2 <= tol:
            STOP = True
            break

    if STOP:
        output = [i, gap1, gap2]
    else:
        output = [-1, gap1, gap2]

    return x, y, output


# ------------------------------------------------------------------------------------
# ------------------------ Coordinate primal-dual algorithm --------------------------
# ------------------------------------------------------------------------------------

@jit(nopython=True, nogil=True, cache=True)
def coo_pd_update_numba(x, y, u, AT, n, steps, sigma, ik):
    """
    Update for the coordinate primal-dual method for basis pursuit
    """
    a = AT[ik]
    tau = steps[ik] / sigma
    t = prox_l1(x[ik] - tau / n * np.dot(a, y), tau / n)
    h = t - x[ik]
    y += u + sigma * (n + 1) * h * a
    u += sigma * h * a
    x[ik] = t

    return x, y, u


def coo_pd_numba(AT, b,  x0, steps, sigma, numb_iter=100, tol=1e-6):
    """
    Coordinate version of the primal-dual algorithm of Pock and Chambolle for problem
    min_x |x|_1 s.t. Ax =b
    
    AT equals to A.T. This is more convenient for the
    algorithm. Notice that AT should have C-contiguous flag. This
    means that A.T will not work, it is better to make a copy
    A.T.copy()
    
    Instead of running a random generator in each iteration, we shuffle indices in advance.

    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria include:
    primal gap (based on the first order condition) and the
    feasibility gap ||Ax-b||.

    """
    n, m = AT.shape

    x = x0.copy()
    u = sigma * (np.dot(AT.T, x0) - b)
    y = u.copy()
    STOP = False

    np.random.seed(0)
    permut = np.arange(n)

    for epoch in range(numb_iter):
        np.random.shuffle(permut)
        for ik in permut:
            #print(ik)
            x, y, u = coo_pd_update_numba(x, y, u, AT, n, steps, sigma, ik)


        f_gap = 1 / sigma * LA.norm(u, ord=np.inf)

        # we don't want to compute s_gap in every iteration, since it
        # requires computing A.T.dot(y). We compute it only if the
        # feasibility gap is already small.
        if f_gap <= tol:
            s_gap = subdif_gap(-np.dot(AT, y), x)
            if s_gap <= tol:
                STOP = True
                break

    if STOP:
        output = [epoch, s_gap, f_gap]
    else:
        f_gap = 1 / sigma * np.sqrt(np.dot(u, u))
        s_gap = subdif_gap(-np.dot(AT, y), x)
        output = [-1, s_gap, f_gap]

    return x, y, output


# ------------------------------------------------------------------------------------
# ------------------------ Block-coordinate primal-dual algorithm --------------------
# ------------------------------------------------------------------------------------


# block-coordinate update
@jit(nopython=True, nogil=True, cache=True)
def coo_block_pd_update_numba(x, y, u, AT, n_block, dim_block, steps, sigma, ik):
    """
    Update for block-coordinate primal-dual method for basis pursuit problem

    n_block : number of blocks
    dim_block: dimension of one block (we assume that all blocks have the same dimension)
    steps: array of inverse operator norms for blocks A[i]
    sigma: dual stepsize. This is the only parameter that influence convergence
    ik: number from 0 to n_block; defines which block to choose.
    """
    block0 = ik * dim_block
    block1 = (ik + 1) * dim_block
    x_block = x[block0: block1].copy()

    # Ai = A[:, block0: block1]
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



def coo_block_pd_numba(AT, b,  x0, steps, sigma, numb_iter=100, tol=1e-6):
    """
    Block-coordinate version of primal-dual algorithm of Pock and Chambolle for problem
    min_x |x|_1 s.t. Ax =b

    AT equals to A.T. This is more convenient for the
    algorithm. Notice that AT should have C-contiguous flag. This
    means that A.T will not work, it is better to make a copy
    A.T.copy()

    The number of blocks equals to n diveded over the size of the array steps.
    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria include:
    primal gap (based on the first order condition) and the
    feasibility gap ||Ax-b||.
    """
    n, m = AT.shape
    x = x0.copy()
    u = sigma * (np.dot(AT.T, x0) - b)
    y = u.copy()

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
            
        f_gap = 1 / sigma * LA.norm(u, ord=np.inf)

        # we don't want to compute s_gap in every iteration, since it
        # requires computing A.T.dot(y). We compute it only if the
        # feasibility gap is already small.
        if f_gap <= tol:
            s_gap = subdif_gap(-np.dot(AT, y), x)
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

    return x, y, output




# ------------------------------------------------------------------------------------
# ------ Full variants of the coordinate algorithms. Useful for line profiling -------
# ------------------------------------------------------------------------------------


def coo_block_pd_full(AT, b,  x0, steps, sigma, numb_iter=100, tol=1e-6):
    """
    Block-coordinate version of the primal-dual algorithm of
    Chambolle-Pock for problem min_x |x|_1 s.t. Ax =b The number of
    blocks equals to the length of steps array.

    AT equals to A.T. This is more convenient for the
    algorithm. Notice that AT should have C-contiguous flag. This
    means that A.T will not work, it is better to make a copy
    A.T.copy()
    
    Instead of running a random generator in each iteration, we shuffle indices in advance

    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria include:
    primal gap (based on the first order condition) and the
    feasibility gap ||Ax-b||.
    """
    n, m = AT.shape
    x = x0.copy()
    u = sigma * (AT.T.dot(x0) - b)
    y = u.copy()

    n_block = len(steps)
    dim_block = n // n_block
    STOP = False

    np.random.seed(0)

    # make permutation of all blocks
    permut = np.arange(n_block)
    for epoch in range(numb_iter):
        np.random.shuffle(permut)
        for i in range(n_block):

            ik = permut[i]
            block0 = ik * dim_block
            block1 = (ik + 1) * dim_block
            x_block = x[block0: block1].copy()

            Ai = AT[block0: block1]


            tau = steps[ik] / sigma
            AiTy = np.dot(Ai, y)

            tmp1 = x_block - (tau / n_block) * AiTy
            block_update = prox_l1(tmp1, tau / n_block)

            h = block_update - x_block
            Aih = np.dot(Ai.T, h)

            y += u + sigma * (n_block + 1) * Aih
            u += sigma * Aih
            x[block0:block1] = block_update

        f_gap = 1 / sigma * LA.norm(u, ord=np.inf)
        
        # we don't want to compute s_gap in every iteration, since it
        # requires computing A.T.dot(y). We compute it only if the
        # feasibility gap is already small.
        if f_gap <= tol:
            s_gap = subdif_gap(-np.dot(AT, y), x)
            if s_gap <= tol:
                STOP = True
                break

    if STOP:
        # n_epoch = i // n_block
        output = [epoch, s_gap, f_gap]
    else:
        f_gap = 1 / sigma * np.sqrt(np.dot(u, u))
        s_gap = subdif_gap(-np.dot(AT, y), x)
        # means that the algorithm does not converge within N*n_batch
        # iterations
        epoch = -1
        output = [epoch, s_gap, f_gap]

    return x, y, output


def coo_pd_full(AT, b,  x0, steps, sigma, numb_iter=100, tol=1e-6):
    """
    Coordinate version of primal-dual algorithm of Pock and Chambolle
    for problem min_x |x|_1 s.t. Ax =b

    AT equals to A.T. This is more convenient for the
    algorithm. Notice that AT should have C-contiguous flag. This
    means that A.T will not work, it is better to make a copy
    A.T.copy()
    
    Instead of running a random generator in each iteration, we
    shuffle indices in advance

    Algorithm runs either for numb_iter iteration or when the stopping
    criteria reaches tol accuracy.  The stopping criteria include:
    primal gap (based on the first order condition) and the
    feasibility gap ||Ax-b||.

    """
    n, m = AT.shape

    x = x0.copy()
    u = sigma * (np.dot(AT.T, x0) - b)
    y = u.copy()

    STOP = False
    np.random.seed(0)
    #make permutation of all blocks
    permut = np.arange(n)
    for epoch in range(numb_iter):
        np.random.shuffle(permut)
        for ik in permut:
            a = AT[ik]
            tau = steps[ik] / sigma

            ay = np.dot(a, y)
            t = prox_l1(x[ik] - (tau / n) * ay, tau / n)
            h = t - x[ik]

            u += (sigma * h) * a
            y += u + (sigma * n * h) * a
            x[ik] = t

        f_gap = 1 / sigma *LA.norm(u, ord=np.inf)
        # we don't want to compute s_gap in every iteration, since it
        # requires computing A.T.dot(y). We compute it only if the
        # feasibility gap is already small.
        if f_gap <= tol:
            s_gap = subdif_gap(-np.dot(AT, y), x)
            if s_gap <= tol:
                STOP = True
                break

    if STOP:
        output = [epoch, s_gap, f_gap]
    else:
        f_gap = 1 / sigma * np.sqrt(np.dot(u, u))
        s_gap = subdif_gap(-np.dot(AT, y), x)
        output = [-1, s_gap, f_gap]

    return x, y, output
