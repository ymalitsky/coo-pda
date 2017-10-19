# Basis pursuit problem  \min_x |x|_1 such that Ax = b. This file generates data, has wrapper for running algorithms and makes some experiments.


import numpy as np
import scipy as sp
import scipy.linalg as LA
import scipy.sparse as spr
from scipy.fftpack import dct as dct

from time import process_time, time
from algorithms import *

from different import *


def generate_data(m, n, l_spr, matrix='random', gen=0):
    """
    The function generates some problems for basis pursuit, depending on the given data. 

    The ways of generating matrix depend on the flag matrix. It can be one of the following

    random: A is a random Gaussian matrix

    random_fixed: as before, but with a fixed random generator for reproducible research

    dct proj: extract random m rows from the discrete cosine transform matrix

    wavelet: A is a composition of the gaussian random matrix and an
    overcomplete dictionary of the Haar transform and the discrete
    cosine transform.
    """
    np.random.seed(0)
    w = np.random.uniform(-10, 10, n)
    w[int(l_spr * n):] = 0
    w = np.random.permutation(w)

    if matrix == "random":
        A = np.random.normal(0, 1, (m, n))
      
    elif matrix == "random_fixed":
        np.random.seed(0)
        A = np.random.normal(0, 1, (m, n))
                
    elif matrix == 'dct_proj':
        w = np.random.normal(0, 1, n)
        w[50:] = 0
        w[:100] = np.random.permutation(w[:100])

        # a matrix of the discrete cosine transform. Of course this is
        # not the best way to generate it, but for our purposes this
        # suffices.
        Phi = dct(np.eye(n)).T
        # extract rows from Phi indexed by ind array
        ind = np.random.choice(n,m, replace=False)
        A = Phi[ind] 

    elif matrix == 'wavelet':
        Phi = dct(np.eye(n)).T
        H = haar_matrix(n)
        B = np.random.normal(0, 1, (m, 2*n))
        Z = np.vstack([Phi, H])
        A = B @ Z

    b = A.dot(w)
    return A, b, w


def run_pda(data, N, j=0, display=True, err=1e-6):
    """
    For given data runs the primal-dual algorithm.

    N: the number of iterations
    j: an integer number, based on which the steps are defined.
    display: boolean values to show or not the performance details of the algorithm
    err: the tolerance for the stopping criteria
    """
    A, b, x0, alpha = data
    sigma = 1 / 2**j * alpha
    tau = 2**j * alpha

    cpu1 = process_time()
    elaps1 = time()
    
    x, y, output = pd_basis_pursuit(
        A, b,  x0, sigma, tau, numb_iter=N, tol=err)

    elaps2 = time()
    cpu2 = process_time()
    time_cpu = cpu2 - cpu1
    time_elaps = elaps2 - elaps1

    if display:
        print('--------- PDA for Basis Pursuit -----------------------')
        print("CPU and elapsed time:", time_cpu, time_elaps)
        print("Optimality gap:", output[1])
        print("Feasibility gap:", output[2])
        print("Number of iterations:", output[0])

    return x, y, output + [time_cpu, time_elaps]


def run_coo(data, N, j=0, alg='block', display=True, err=1e-6):
    """
    For given data runs the block-coordinate primal-dual algorithm.

    N: the number of iterations
    j: an integer number, based on which the steps are defined

    alg: can be 'block' or 'coo'. The latter means fully coordinate
    with batch = 1.  At the moment there are possibilities to use
    "block_full" or "coo_full". This is the same implementations of
    the above methods just written in one function. Remove this after
    testing.

    display: boolean values to show or not the performance details of
    the algorithm
    err: the tolerance for the stopping criteria
    """
    A, b, x0,  d_batch, n_batch = data
    AT = A.T.copy()
    sigma = 1 / (2**j) * 1. / n_batch
    cpu1 = process_time()
    elaps1 = time()
    steps = find_Lipsch_const(AT, d_batch)

    if alg == 'block':
        x, y, output = coo_block_pd_numba(
            AT, b,  x0, steps, sigma, numb_iter=N, tol=err)

    elif alg == 'block_full':
        x, y, output = coo_block_pd_full(
            AT, b,  x0, steps, sigma, numb_iter=N, tol=err)

    elif alg == 'coo':
        x, y, output = coo_pd_numba(
            AT, b,  x0, steps, sigma, numb_iter=N, tol=err)

    elif alg == 'coo_full':
        x, y, output = coo_pd_full(
            AT, b,  x0, steps, sigma, numb_iter=N, tol=err)

    elaps2 = time()
    cpu2 = process_time()
    time_cpu = cpu2 - cpu1
    time_elaps = elaps2 - elaps1

    if display:
        print(
            '--------- {}-PDA for Basis Pursuit -----------------------'.format(alg))
        print("CPU and elapsed time:", time_cpu, time_elaps)
        print("Optimality gap:", output[1])
        print("Feasibility gap:", output[2])
        print("Number of iterations:", output[0])

    return x, y, output + [time_cpu, time_elaps]


def find_Lipsch_const(AT, d_batch):
    """
    For the matrix A of the shape (m,n), given in the transposed form
    for simplicity, and the given batch size, the function finds the
    steps that are the inverse Lipschitz constants of the blocks
    A1,..., Ak, where k = n//d_batch
    """
    n = AT.shape[0]
    n_batch = n // d_batch
    
    # if the dimension of a block is larger than 1:
    if d_batch > 1:
        # rewrite in a better way
        ls = []
        for i in range(n_batch):
            S = AT[i * d_batch: (i + 1) * d_batch]
            L = LA.eigh(S.dot(S.T), eigvals_only=True, eigvals=(d_batch - 1, d_batch - 1))
            ls.append(L)
        Lipsch_const = np.array(ls)
        
    # if the dimension of a block is 1:
    else:
        Lipsch_const = LA.norm(AT, axis=1)**2

    steps = 1. / Lipsch_const
    return steps


def make_experiment(experiments, N, filename, matrix='random_fixed', tol=1e-6):
    """
    Make experiments for different test problems.

    N:  number of epochs
    filename: the path to the file where the results should be written
    tol: the desired accuracy for the stopping criteria
    """
    for i, exp_i in enumerate(experiments):
        m, n, d_batch, l_spr = exp_i
        gen = 0

        i_low = 0
        i_up = 12
        i_range = range(i_low, i_up)

        A, b, w = generate_data(m, n, l_spr, matrix)
        #A, b, w = generate_data(m, n, l_spr, 'random_fixed')
        
        x0 = np.zeros(n)
        y0 = A.dot(x0) - b
        L = LA.eigh(A.dot(A.T), eigvals_only=True, eigvals=(m-1,m-1))
        alpha = 1 / np.sqrt(L)

        n_batch = n // d_batch

        data_pda = [A, b, x0, alpha]
        data_coo = [A, b, x0, 1, n]

        data_block = [A, b, x0, d_batch, n_batch]
        
        
        pda_ls = []
        coo_block_ls = []
        coo_ls = []
        
        for ik in i_range:
            x1, y1, output1 = run_pda(
                data_pda, N, j=ik, display=False, err=1e-6)

            x2, y2, output2 = run_coo(
                data_block, N, alg='block', j=ik, display=False, err=1e-6)

            x3, y3, output3 = run_coo(
                data_coo, N, alg='coo', j=ik, display=False, err=1e-6)

            
            if output1[0] != -1:
                output1.append(LA.norm(x1))
                pda_ls += [[ik] + output1]

            if output2[0] != -1:
                output2.append(LA.norm(x2))
                coo_block_ls += [[ik] + output2]

            if output3[0] != -1:
                output3.append(LA.norm(x3))
                coo_ls += [[ik] + output3]


        
        if pda_ls != []:
            make_table(i, exp_i, pda_ls, 'PDA', filename)
        else:
            print("PDA: does not converge")

        if coo_block_ls != []:
            make_table(i, exp_i, coo_block_ls, 'Block-Coo', filename)
        else:
            print("Block-Coo: does not converge")

        if coo_ls != []:
            make_table(i, exp_i, coo_ls, 'Coo', filename)
        else:
            print("Coo no: does not converge")
            


if __name__ == "__main__":
  
    experiment1 = [(1000, 4000, 50,  0.05), (2000, 8000, 50, 0.05), (4000, 16000, 50, 0.05)]
    #experiment1 = [(1000, 4000, 50,  0.05)]

    filename ='results/random_fixed_bp-1.org'
    #filename ='results/dct_proj_bp-2.org'
    
    N = 1000
    make_experiment(experiment1, N, filename, matrix='random_fixed', tol=1e-6)
    #make_experiment(experiment1, N, filename, matrix='dct_proj', tol=1e-6)

