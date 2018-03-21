"""
Connection matrix.
"""

from itertools import combinations
import math
import numpy as np


def n_combinations(n, r):
    """
    >>> n_combinations(5, 2)
    10
    """
    assert n >= r
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def matrix_0(d, r):
    """
    Returns a connection matrix for:
    d : number of dimensions in the input matrix
    r : how many of `d` to choose
    """
    n = n_combinations(d, r)
    a = np.zeros((d, n), dtype=np.byte)
    for i, combo in enumerate(combinations(range(0, d), r)):
        for row in combo:
            a[row][i] = 1
    return a


def matrix(d, r0, r1):
    assert r0 < r1
    n0 = n_combinations(d, r0)
    n1 = n_combinations(d, r1)
    a = np.zeros((n0, n1), dtype=np.byte)

    dict_0 = {}
    for i, combo_0 in enumerate(combinations(range(0, d), r0)):
        dict_0[combo_0] = i

    for i1, combo_1 in enumerate(combinations(range(0, d), r1)):
        for combo_0 in combinations(combo_1, r0):
            i0 = dict_0[combo_0]
            a[i0][i1] = 1
    return a





