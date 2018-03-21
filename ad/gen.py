"""
Data generation.
"""

import numpy as np

s_min = -2
s_max = +2


def s_breaks(b):
    """
    :param b: number of breakpoints (must be odd)
    :return: an array spaced between [s_min, s_max], inclusive.

    If b == 1: [0]
    If b == 3: [-2, 0, 2]
    If b == 5: [-2, -1, 0, 1, 2]
    If b == 9: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    """
    return np.linspace(s_min, s_max, num=b)


def s_gaps(g):
    """
    :param g: number of gaps
    :return: a 2 x k array of gaps. Row 0 has start coordinates. Row 1 has
    end coordinates.

    Example return value:

    np.array([
      [0.6555, -0.9614, 1.3497],
      [0.869,  -0.7522, 1.4861]
    ])
    """
    c = np.random.uniform(s_min, s_max, g)
    w = np.abs(np.random.normal(0.15, 0.1, g))
    return np.vstack([c - w, c + w])


def random_s(gaps, max_iter=20):
    """
    :param gaps: gaps in s (from `s_gaps`)
    :param max_iter: maximum iterations
    :return: a random sample of s (scalar double)
    """
    i = 0
    s = None
    while not s and i < max_iter:
        s = random_s_base(gaps)
        i += 1

    if i == max_iter:
        raise(RuntimeError('exceeded max iterations {}'.format(i)))
    else:
        return s


def random_s_base(gaps):
    """
    :param gaps: gaps in s (from `s_gaps`)
    :return: a random sample of s (scalar double) or `None`
    """
    x = np.random.uniform(s_min, s_max, 1)
    for i in range(0, gaps.shape[1]):
        if gaps[0][i] < x < gaps[1][i]:
            return None
    return x[0]


def func_matrix(b, d):
    """
    :param b: number of breakpoints
    :param d: number of observable dimensions
    :return: a `b+1` x `d` array
    """
    a = np.random.normal(0, 0.5, (b + 1) * d)
    return a.reshape((b + 1, d))


def memo_func(breaks, matrix):
    """
    :param breaks:
    :param matrix:
    :return: an array of length `2 * b`
    """
    b = len(breaks)
    a = np.empty((2 * b, matrix.shape[1]))
    i = 0
    for s in breaks:
        a[2 * i] = s * matrix[i]
        a[2 * i + 1] = s * matrix[i + 1]
        i += 1
    return a


def observation(s, breaks, matrix, memo):
    """
    Returns an observation using memoization, based on the parameter s.
    """
    b = len(breaks)
    if b == 0 or s <= breaks[0]:
        return s * matrix[0]
    for i in range(1, b + 1):
        if i == b or s <= breaks[i]:
            j = (i - 1) * 2
            return memo[j] + s * matrix[i] - memo[j + 1]


def noise(d):
    return np.random.normal(0, 0.05, d)


def observations(n, d, b, g):
    """
    :param n: number of observations
    :param d: number of observable dimensions
    :param b: number of breakpoints
    :param g: number of gaps
    :return: a n x d array of observations
    """
    xs = np.empty((n, d), dtype=np.float32)
    breaks = s_breaks(b)
    gaps = s_gaps(g)
    matrix = func_matrix(b, d)
    memo = memo_func(breaks, matrix)
    for i in range(0, n):
        s = random_s(gaps)
        o = observation(s, breaks, matrix, memo)
        o += noise(d)
        xs[i] = o
    return xs
