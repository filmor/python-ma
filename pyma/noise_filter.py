import scipy.linalg as la
import numpy as np
import math

# Implementation of the algorithms given in Doclo1998

# TODO: Implement multi-channel variant

def to_vector(m):
    assert len(m.shape) == 2 and m.shape[0] == m.shape[1]
    off_diagonal_indices = np.tril_indices_from(m, -1)
    diag = np.diag(m)
    return np.concatenate((diag, m[off_diagonal_indices]))

def from_vector(x):
    # Solution to the equation len(x) = n * (n + 1) / 2
    n = int((math.sqrt(len(x) * 8 + 1) - 1) / 2)
    result = np.zeros((n, n))
    result[np.tril_indices_from(result, -1)] = x[n:]
    result += result.transpose()
    result[np.diag_indices_from(result)] = x[:n]
    return result

def anti_diag(X, n=0):
    N, M = X.shape
    lower = max(n + 1 - M, 0)
    upper = min(n, N - 1)
    for i in range(lower, upper + 1):
        yield X[i, n - i]

def get_hankel_matrix(y, L=None):
    N = len(y)
    if L is None:
        L = N // 2 + 1
    M = N - L + 1
    return la.hankel(y)[:L,:M]

def get_values_from_hankel_matrix(m):
    return np.array([
        np.mean(list(anti_diag(m, i)))
        for i in range(sum(m.shape) - 1)
        ])

def data_to_hankel_matrix(y):
    if len(y.shape) > 2:
        raise ArgumentError
    elif len(y.shape) == 2:
        H = np.concatenate([
                get_hankel_matrix(m).transpose()
                for m in y.transpose()
            ]).transpose()
    else:
        H = get_hankel_matrix(y)
    return H

def data_from_hankel_matrix(y, X):
    if len(y.shape) == 2:
        size = y.shape[0] // 2
        return np.array([
                get_values_from_hankel_matrix(X[:,n:n+size])
                for n in range(0, X.shape[1], size)
            ]).transpose()
    else:
        return get_values_from_hankel_matrix(X)


def filter_step(y, modifier):
    if type(modifier) is int:
        p = modifier
        def func(s):
            s[p:] = 0
            return s
        modifier = func

    H = data_to_hankel_matrix(y)

    u, s, vh = la.svd(H, full_matrices=False)
    s = modifier(s)
    X = np.dot(u, np.dot(la.diagsvd(s, u.shape[0], vh.shape[0]), vh))
    return data_from_hankel_matrix(y, X)


def classical_se_algorithm(y, n, p):
    for i in range(n):
        y = filter_step(y, p)
    return y

def ise_algorithm(x, l, p, n):
    for i in range(l):
        y = np.copy(x)
        x = np.zeros_like(y)
        
        for j in range(p):
            s = filter_step(y, n)
            x += s
            y -= s
    return x

def run_on_flattened_data(data, algo, *args, **kwargs):
    res = []
    for d in data:
        flattened = np.array(tuple(to_vector(m) for m in d)) 
        result = algo(flattened, *args, **kwargs)
        res.append(np.array(tuple(from_vector(m) for m in result)))
    return np.array(res)

def classical_matrix_se_algorithm(data, **kwargs):
    return run_on_flattened_data(data, classical_se_algorithm, **kwargs)

def matrix_ise_algorithm(data, **kwargs):
    return run_on_flattened_data(data, ise_algorithm, **kwargs)

def matrix_minimum_variance(data, n):
    algo = classical_se_algorithm

    noisy = data[1:][:,n:,:,:]

    noise = []

    for d in noisy:
        x = np.array([to_vector(m) for m in d])
        H = data_to_hankel_matrix(x)
        _, s, _ = la.svd(H, full_matrices=False)
        noise.append(s)

    noise = np.mean(noise)

    def func(s):
        n = np.ones_like(s) * noise
        res = s - n ** 2 / s
        return res

    return run_on_flattened_data(data, classical_se_algorithm, p=func, n=10)

def minimum_variance(x, n):
    """n: noise limit"""
    H = data_to_hankel_matrix(x[n:])
    u, s, vh = la.svd(H, full_matrices=False)
    noise = s * 10

    def modifier(s):
        n = np.zeros_like(s)
        n[:noise.shape[0]] = noise
        res = s - n ** 2 / s
        return res

    return filter_step(x, modifier)

