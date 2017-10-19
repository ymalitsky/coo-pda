# Wrappers for the algorithms.

import numpy as np
import scipy.linalg as LA
from time import process_time, time
from scipy.fftpack import dct as dct
from bp_noise_algorithms import *
from utils import haar_matrix
import matplotlib.pyplot as plt

def generate_data(m, n, l_spr, matrix='uniform', noise='normal', var=1.0, gen=0):
    """
    The function generates some problems for basis denoising, depending on the given data. 

    The ways of generating matrix depend on the flag matrix. It can be one of the following
    normal: all entries of A are random variables from N(0,1)
    uniform: all entries of A are random variables  from U(-1,1)

    low_rank: A is the product of two gaussian matrices with shapes (m, m/2) and (m/2,n)

    wawelet: A is a composition of the gaussian random matrix and an
    overcomplete dictionary of the Haar transform and the discrete
    cosine transform.

    The noise is generated using the following flags:
    normal: gaussian vector from N(0, var)
    round_off: rounds off the var digits after comma.
    uniform: random vector from the uniform distribution U(-var, var)
    impulse: add noise of 1 to random m/10 coordinates

    var: is the parameter for the randomly generated noise (variance)
    or number of digits for rounding off.
    """
    np.random.seed(gen)
    w = np.random.uniform(-10, 10, n)
    w[l_spr:] = 0
    w = np.random.permutation(w)

    if matrix == 'low_rank':
        rank = m//2
        AL = np.random.normal(0, 1, (m, rank))
        AR = np.random.normal(0, 1, (rank, n))
        A = AL.dot(AR)
    elif matrix == 'normal':
        A = np.random.normal(0, 1, (m, n))

    elif matrix == 'uniform':
        A = np.random.uniform(-1, 1, (m, n))
        
    elif matrix == 'wavelet':
        Phi = dct(np.eye(n)).T
        H = haar_matrix(n)
        B = np.random.normal(0, 1, (m, 2*n))
        Z = np.vstack([Phi, H])
        A = B @ Z
        
    b_true = A.dot(w)
    
    if noise == "normal":
        nu = np.random.normal(0,1,m)*var
        
    elif noise == 'round-off':
        nu = np.round(b_true, var) - b_true
        #nu = np.zeros(m)
        
    elif noise == 'uniform':
        nu =  np.random.uniform(-1,1,m)*var
    elif noise == 'impulse':
        nu =  np.zeros(m)
        nu[:m//10] = 1.
        nu = np.random.permutation(nu)
        
    b = b_true + nu

    return A, b, w


def run_pda(data, N, j=0, display=True, err=1e-6):
    """
    For given data runs the primal-dual algorithm.

    N: the number of iterations
    j: an integer number, based on which the steps are defined.
    display: boolean values to show or not the performance details of the algorithm
    err: the tolerance for the stopping criteria
    """
    A, b, w, x0, alpha = data
    sigma = 1 / 2**j * alpha
    tau = 2**j * alpha

    cpu1 = process_time()
    elaps1 = time()
    

    x, y, output, ls_error, ls_feas = pd_basis_pursuit(
        A, b,  w, x0, sigma, tau, numb_iter=N, tol=err)

    elaps2 = time()
    cpu2 = process_time()
    time_cpu = cpu2 - cpu1
    time_elaps = elaps2 - elaps1

    if display:
        print('--------- PDA for Basis Pursuit -----------------------')
        print("CPU and elapsed time:", time_cpu, time_elaps)
        print("Optimality gap:", output[1])
        print("Feasibility gap:", output[2])
        #print("Gaps:", gap(A,b,x), gap(A,b, w))
        print("Energy", LA.norm(x,1), LA.norm(w,1))
        print("Number of iterations:", output[0])
    return x, y, output + [time_cpu, time_elaps], ls_error, ls_feas


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
    A, b, w, x0,  d_batch, n_batch = data
    AT = A.T.copy()
    sigma = 1 / (2**j) * 1. / n_batch
    # sigma = 1 / (2**j)
    cpu1 = process_time()
    elaps1 = time()
    steps = find_Lipsch_const(AT, d_batch)

    if alg == 'block':
        x, y, output, ls_error, ls_feas = coo_block_pd_numba(
            AT, b, w, x0, steps, sigma, numb_iter=N, tol=err)

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

    return x, y, output + [time_cpu, time_elaps], ls_error, ls_feas

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


def make_experiment(experiments, N, folder, tol=1e-6, d_batch=50):
    """
    Make plots for different stepsizes

    N:  number of epochs
    folder: the name of the folder, where the plots should be written
    tol: the desired accuracy for the stopping criteria    
    """
    for i, exp_i in enumerate(experiments):
        m, n, l_spr, matrix_, noise_, var_ = exp_i
        gen = 0

        i_low = 0
        i_up = 25
        i_range = range(i_low, i_up)

        A, b, w = generate_data(m, n, l_spr, matrix=matrix_, noise=noise_, var=var_, gen=1)

        x0 = np.zeros(n)
        n_batch = n // d_batch
        data_block = [A, b, w, x0, d_batch, n_batch]
        
        min_err = []
        n_fig= (i_up-i_low)
        fig, ax = plt.subplots(nrows=n_fig, ncols=2, figsize=(12,2*n_fig))
                               
        for ik in i_range:
            x2, y2, output2, ls_error, ls_feas = run_coo(
                data_block, N, alg='block', j=ik, display=False, err=tol)
            min_err.append([ik,min(ls_error)])
            row = ik

            ax[row, 0].plot(ls_error,'b')    
            ax[row, 1].plot(ls_feas,'g')
            ax[row,0].set_yscale('log')
            ax[row,1].set_yscale('log')
            ax[row,0].set(title=ik)
        fig.savefig('{0}/exp-{1}.pdf'.format(folder, i), bbox_inches='tight')
        fig.clf()

        sorted_error=sorted(min_err, key=lambda item: item[1])
        print("Experiment ", i)
        print(sorted_error[:5])
        print("----------------------------------------------")

         

if __name__ == "__main__":
    m, n = 200, 400
    A, b, w = generate_data(m, n, 50, matrix='uniform', noise='normal')
    B = A.T.dot(A)
    c = A.T.dot(b)
    # A, b, w = generate_data(m, n, 0.01, 'dct')
    x0 = np.zeros(n)
    #y0 = A.dot(x0) - b
    L = LA.eigh(A.dot(A.T), eigvals_only=True, eigvals=(m - 1, m - 1))
    #L2 = np.max(LA.eigh(A.dot(A.T))[0])
    
    alpha = 1 / np.sqrt(L)
    beta = 1/L
    #alpha = 1e-6
    d_batch = 10
    n_batch = n // d_batch
    #steps = find_Lipsch_const(A, d_batch)
    # steps = [alpha**2]

    data_pda = [A, b, w, x0, alpha]
    data_pda2 = [B, c, x0, beta]
    data_coo = [A, b, x0, 1, n]
    data_block = [A, b, w, x0, d_batch, n_batch]

    ik = 7
    sigma = 1 / 2**ik * alpha
    # sigma = 1
    tau = 2**ik * alpha

    N = 100
    if 1 <  2:
        ans1 = run_pda(data_pda,  N, j=4)
        ans2 = run_coo(data_block, N, alg='block', j=8)
   
    experiment1 = [
        (1000, 4000, 50,  'normal', 'uniform', 0.1),
        (1000, 4000, 50,  'normal', 'uniform', 1.0),
        (1000, 4000, 50,  'low_rank', 'uniform', 0.1),
        (1000, 4000, 50,  'low_rank', 'normal', 1)
    ]

    
    #N = 50000
    folder = 'results/experiment1'

    N = 1000
    make_experiment(experiment1, N, folder, tol=1e-6, d_batch=50)
    print('lala')
