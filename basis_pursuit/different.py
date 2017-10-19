# different useful functions

import numpy as np

import scipy.sparse as spr
import scipy.linalg as LA
from scipy.fftpack import dct as dct
from tabulate import tabulate
import line_profiler


def haar_matrix(size):
    """
    Code for the Haar matrix -- the matrix of the Haar transform.
    """
    level = int(np.ceil(np.log2(size)))
    H = np.array([1.])[:, None]
    NC = 1. / np.sqrt(2.)
    LP = np.array([1., 1.])[:, None] 
    HP = np.array([1., -1.])[:, None]
    for i in range(level):
        H = NC * np.hstack((np.kron(H, LP), np.kron(np.eye(len(H)),HP)))
    H = H.T
    return H

def make_table(i, exp_i, output_data, alg, filename):
    with open(filename, 'a') as outputfile:
        outputfile.write(
            '* Exper {0},    m={1},  n={2}, d_block={3}, spr={4} \n'.format(i, *exp_i))
        outputfile.write('** ' + alg)
        outputfile.write('\n')
        headers = ['i', 'epoch',  'p_gap',
                   'f_gap', 'cpu', 'elapsed', 'energy']
        best_tuple = min(output_data, key=lambda item: item[1])
        group_data = list(zip(*output_data))
        table = [[headers[i], best_tuple[i]] + list(
            group_data[i]) for i in range(7)]

        outputfile.write(tabulate(table, tablefmt='orgtbl'))
        outputfile.write('\n')

def my_line_profiler(f, data_string):
    l = line_profiler.LineProfiler()
    l.add_function(f)
    l.run(data_string)
    l.print_stats()
