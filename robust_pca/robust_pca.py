#  This program compares the primal-dual and the coordinate
#  primal-dual algorithms for Robust Principal Component Analysis
#  problem for different data.  RPCA is the following problem:
#  \min_{L,S} |L|_* + delta * |S|_1 such that L + S = M.

import numpy as np
import scipy.linalg as LA
from time import process_time, time
from sklearn.decomposition import randomized_svd
from tabulate import tabulate



def prox_L_exact(X, q):
    """
    Compute prox operator for the nuclear norm. Works for any matrix
    X. Should be fast enough, especially when m != n, since svd
    does not compute full matrices
    """
    m, n = X.shape
    U, s, V = LA.svd(X, full_matrices=False)
    #print(q, np.abs(s[-5:]))
    soft_thr = np.maximum(np.abs(s) - q, 0) * np.sign(s)
    X_new = (U * soft_thr).dot(V)
    return X_new



def prox_L_random(X, q, k=70):
    """
    Compute prox operator for the nucler norm using the ranomized SDV from sklearn.decomposition
    Use k=70 eigenvalues
    """
    m, n = X.shape
    U, s, V = randomized_svd(X, k)
    soft_thr = np.maximum(np.abs(s) - q, 0) * np.sign(s)
    X_new = (U * soft_thr).dot(V)
    return X_new


def prox_L(X, q):
    return prox_L_random(X,q)
    #return prox_L_exact(X,q)


def prox_S(X, rho):
    """
    Compute prox operator for l1 norm. Standard soft thresholding
    """
    return X + np.clip(-X, -rho, rho)


def nuclear_norm(X):
    s = LA.svd(X, compute_uv=0)
    return LA.norm(s, 1)


def energy(L, S, delta):
    """
    Compute |L|_* + delta * |S|_1
    """
    return nuclear_norm(L) + delta * np.sum(np.abs(S))


def pd_rbca(M, delta,  L0, S0, Y0, sigma, tau, tol=1e-6, numb_iter=100):
    """
    Primal-dual algorithm of Pock-Chambolle for Robust PCA
    """
    L, S, Y = L0.copy(), S0.copy(), Y0.copy()
    M_norm = LA.norm(M)
   
    STOP = False
    for i in range(numb_iter):

        L1 = prox_L(L - tau * Y, tau)
        S1 = prox_S(S - tau * Y, tau * delta)
        L_bar = 2 * L1 - L
        S_bar = 2 * S1 - S

        Y += sigma * (L_bar + S_bar - M)

        # we start to check the stopping criteria after 10 iterations
        # primal_gap bounds the distance between two subdifferentials of |X|_*
        # and delta*|X|_1
        if i >= 10:
            primal_gap = 1. / tau * np.max(np.abs(S - S1 - L + L1))/M_norm
            feas_gap = np.max(np.abs(S + L - M))/M_norm

            if primal_gap < tol and feas_gap < tol:
                # print("Number of epochs:", i)
                STOP = True
                break

        L, S = L1, S1

    if STOP:
        output = [i, primal_gap, feas_gap]
    else:
        output = [-1, primal_gap, feas_gap]

    return L, S, output


def run_pda(data, N, j=0, display=True, err=1e-6):
    cpu1 = process_time()
    elaps1 = time()
    M, delta, L0, S0, Y0, alpha = data
    sigma = 1 / 2**j * alpha
    tau = 2**j * alpha

    L, S, gap_iter = pd_rbca(
        M, delta,  L0, S0, Y0, sigma, tau, tol=err, numb_iter=N)
    elaps2 = time()
    cpu2 = process_time()
    time_cpu = cpu2 - cpu1
    time_elaps = elaps2 - elaps1
    # print output
    if display:
        print('--------- PDA for Robust PCA -----------------------')
        print("Primal energy and complement:", energy(L, S, delta),
              energy(M, M - L, delta))
        print("Primal gap:", gap_iter[1])
        print("Feasibility gap:", gap_iter[2])
        print("CPU total time and elapsed:", time_cpu, time_elaps)
        print("Number of iterations:", gap_iter[0])
    return L, S, gap_iter + [time_cpu, time_elaps]


def coo_pd_rbca(M, delta, L0, S0, sigma, steps, tol=1e-6, numb_iter=100):
    """
    Coordinate  primal-dual algorithm for Robust PCA
    """
    L, S = L0.copy(), S0.copy()
    Y = sigma * (L + S - M)
    M_norm = LA.norm(M)
    
    svd = 0
    l1 = 0
    
    STOP = False

    np.random.seed(0)
    one_or_zero = np.random.randint(0,2, numb_iter)
    for i in range(numb_iter):

        #ik = random.randint(0, 1)
        ik = one_or_zero[i]
        tau = steps[ik] / sigma

        if ik == 0:
            svd += 1
            L1 = prox_L(L - tau / 2 * Y, tau / 2)
            # if svd > -1:
            #     L1 = prox_L(L - tau / 2 * Y, tau / 2, k=60)
            # else:
            #     L1 = prox_L_fixed(L - tau / 2 * Y, tau / 2)
                
            Y += sigma * ((3 * L1 - 2 * L) + S - M)
            subgrad1 = L - L1 - tau / 2 * Y
            L = L1

        else:
            l1 += 1
            S1 = prox_S(S - tau / 2 * Y, tau * delta / 2)
            Y += sigma * ((3 * S1 - 2 * S) + L - M)
            subgrad2 = S - S1 - tau / 2 * Y
            S = S1

        # we start to check the stopping criteria after 100 iterations
        # primal_gap bounds the distance between two subdifferentials of |X|_*
        # and delta*|X|_1
        if i >= 20:
            primal_gap = 2 / tau * np.max(np.abs(subgrad1 - subgrad2))/M_norm
            # multiple over 2/tau because subgrad \in tau/2 * subdiff

            #feas_gap = LA.norm(S + L - M)
            feas_gap = np.max(np.abs(S+L-M))/M_norm
            if primal_gap < tol and feas_gap < tol:
                # print("Number of epochs:", i // 2)
                STOP = True
                break

    if STOP:
        output = [i, primal_gap, feas_gap]
    else:
        output = [-1, primal_gap, feas_gap]
    print("number of svd:", svd, "number of prox_l1", l1)
    return L, S, output, svd, l1


def run_coo(data, N, j=0, display=True, err=1e-6):
    elaps1 = time()
    cpu1 = process_time()

    M, delta, L0, S0, steps = data
    sigma = 1 / 2**j
    L, S, gap_iter, svd, l1 = coo_pd_rbca(
        M, delta,  L0, S0,  sigma, steps, tol=err, numb_iter=N)

    elaps2 = time()
    cpu2 = process_time()
    time_cpu = cpu2 - cpu1
    time_elaps = elaps2 - elaps1
    # print output
    if display:
        print('--------- Coo_PDA for Robust PCA -----------------------')
        print("Primal energy and complement:", energy(L, S, delta),
              energy(M, M - L, delta))
        print("Primal gap:", gap_iter[1])
        print("Feasibility gap:", gap_iter[2])
        print("CPU total time and elapsed:", time_cpu, time_elaps)
        print("Number of iterations:", gap_iter[0])
        print("Number of SVD:", svd)
        print("Number of l1-steps:", l1)
    return L, S, gap_iter + [time_cpu, time_elaps]


def generate_M(n1, n2, r, dns, gen=0):
    """
    For given dimensions n1, n2, given rank r and denisty dns the
    function generates the matrix M, and the true matrices L and S.  

    gen: random generator
    """
    np.random.seed(gen)
    delta = 1 / np.sqrt(max(n1, n2))
    L1 = np.random.randn(n1, r)
    L2 = np.random.randn(r, n2)
    L = L1.dot(L2)

    s = np.random.uniform(-500, 500, n1 * n2)
    nonzero = int((dns * n1 * n2))
    s[nonzero:] = 0
    s = np.random.permutation(s)
    S = s.reshape((n1, n2))
    e1 = nuclear_norm(L + S)
    e2 = energy(L, S, delta)
    print(e1, e2, e1 > e2)
    return L + S, L, S


def residual(L_true, L):
    return LA.norm(L_true-L)/LA.norm(L_true)
        
def make_experiment(experiments, N, filename):
    """
    Compare both methods for RPCA for different problems and different stepsizes.
    """
    for i, exp_i in enumerate(experiments):
        n1, n2, r = exp_i
        gen = 0
        M, L_true, S_true = generate_M(n1, n2, r, 0.05, gen=gen)
        delta = 1 / np.sqrt(max(n1, n2))

        true_energy = energy(L_true, S_true, delta)
        possible_energy = nuclear_norm(M)

        i_low = 6
        i_up = 9
        i_range = range(i_low, i_up)

        if possible_energy < true_energy:
            print("Data for experiment ", exp_i, " is not correct")
        else:
            L0 = M
            S0 = np.zeros((n1, n2))
            Y0 = L0 + S0 - M
            steps = 1. / np.array([1, 1])  # for coo_pda
            alpha = 1. / np.sqrt(2)  # for pda
            data_pda = [M, delta, L0, S0, Y0, alpha]
            data_coo = [M, delta, L0, S0, steps]

            pda_ls = []
            coo_ls = []
            path = 'results/{}'.format(filename)
            
            for ik in i_range:
                    
                L1, S1, output1 = run_pda(data_pda, N, j=ik, display=False)
                
                if output1[0] != -1:
                    output1.append(energy(L1, S1, delta))
                    pda_ls += [[ik] + output1 + [residual(L_true, L1)]]
                    
                L2, S2, output2 = run_coo(data_coo, N, j=ik, display=False)
                
                if output2[0] != -1:
                    output2.append(energy(L2, S2, delta))
                    coo_ls += [[ik] + output2 + [residual(L_true, L2)]]
                    
            if pda_ls != []:
                make_table(i, exp_i, pda_ls, 'PDA', path)

            if coo_ls != []:
                make_table(i, exp_i, coo_ls, 'Coo', path)


def make_table(i, exp_i, output_data, alg, path):
    with open(path, 'a') as outputfile:
        outputfile.write(
            '* Exp {0},    n1={1},  n2={2}, r={3} \n'.format(i, *exp_i))
        outputfile.write('** ' + alg)
        outputfile.write('\n')
        headers = ['ik', 'epoch', 'p_gap',
                   'f_gap', 'cpu', 'elapsed', 'energy', 'residual']
        best_tuple = min(output_data, key=lambda item: item[1])
        group_data = list(zip(*output_data))
        table = [[headers[i], best_tuple[i]] + list(
            group_data[i]) for i in range(8)]

        outputfile.write(tabulate(table, tablefmt='orgtbl'))
        outputfile.write('\n')

if __name__ == "__main__":
    # before running choose which version of SVD you would like to
    # use: exact or randomized. You can choose one of those in lines
    
    experiment = [
        (1000, 500, 20), (1500, 500, 20), (2000, 500, 50), (1000, 1000, 50), (2000, 1000, 50)
    ]
    
    filename='experiment1/rb-svd-1.org'
    N = 300
    make_experiment(experiment, N, filename)
    
