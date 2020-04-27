"""
This module provides an experiment on random graphs and their rank.
"""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

n = 30

def get_random_matrix(p):
    A_raw = np.random.rand(n, n)
    A_shifted = A_raw - 0.5 + p
    A_integer = np.matrix.round(A_shifted)
    A = A_integer

    return A

def get_random_rank(p):
    A = get_random_matrix(p)
    rank = np.linalg.matrix_rank(A)

    return rank

def elucidate_random_graph_rank():
    probabilities = np.linspace(0, 1, 1 + 1000)
    ranks = [ get_random_rank(p) for p in probabilities ]

    print('ranks\n', ranks)
    plt.plot(ranks)
    plt.show()

    return ranks

if __name__ == "__main__":
    elucidate_random_graph_rank()
