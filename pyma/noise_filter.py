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

def get_hankel_matrix(y):
    l = len(y)
    L = l // 2 + 1
    M = L - (0 if l % 2 == 1 else 1)
    return la.hankel(y)[:L,:M]

def get_values_from_hankel_matrix(m):
    return np.array([
                        np.mean(list(anti_diag(m, i)))
                        for i in range(sum(m.shape) - 1)
                     ])

def filter_step(y, p):
    if len(y.shape) > 2:
        raise ArgumentError
    elif len(y.shape) == 2:
        H = np.concatenate([
                get_hankel_matrix(m).transpose()
                for m in y.transpose()
            ]).transpose()
    else:
        H = get_hankel_matrix(m)

    u, s, vh = la.svd(H, full_matrices=False)
    s[p:] = 0
    X = np.dot(u, np.dot(la.diagsvd(s, u.shape[0], vh.shape[0]), vh))

    if len(y.shape) == 2:
        size = y.shape[0] // 2
        return np.array([
                get_values_from_hankel_matrix(X[:,n:n+size])
                for n in range(0, X.shape[1], size)
            ]).transpose()
    else:
        return get_values_from_hankel_matrix(X)

def classical_se_algorithm(y, n, p):
    for i in range(n):
        y = filter_step(y, p)
    return y

def ise_algorithm(x, l, p):
    for i in range(l):
        y = np.copy(x)
        x = np.zeros_like(y)
        
        for j in range(p):
            s = filter_step(y, 1)
            x += s
            y -= s
    return x

def classical_matrix_se_algorithm(d, n, p):
    flattened = np.array(tuple(to_vector(m) for m in d)) 
    result = classical_se_algorithm(flattened, n, p)
    return np.array(tuple(from_vector(m) for m in result))

def matrix_ise_algorithm(d, l, p, symmetric=True):
    flattened = np.array(tuple(to_vector(m) for m in d)) 
    result = ise_algorithm(flattened, l, p)
    return np.array(tuple(from_vector(m) for m in result))

