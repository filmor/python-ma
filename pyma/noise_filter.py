import scipy.linalg as la
import numpy as np

# Implementation of the algorithms given in Doclo1998

def anti_diag(X, n=0):
    N = X.shape[0]
    lower = max(n + 1 - N, 0)
    upper = min(n, N - 1)
    for i in range(lower, upper + 1):
        yield X[i, n - i]

def filter_step(y, p=4):
    if p is None:
        p = len(y) - 1
    H = la.hankel(y)[:len(y) / 2 + 1,:len(y) / 2 + 1]
    u, s, vh = la.svd(H)
    s[p:] = 0
    X = np.dot(u, np.dot(np.diag(s), vh))
    return np.array([
        np.mean(list(anti_diag(X, i))) for i in range(len(y))
        ])

def classical_se_algorithm(y, n, p):
    for i in range(n):
        y = filter_step(y, p)
    return y

def ise_algorithm(x, l, p):
    for i in range(l):
        y = x[:]
        x = np.zeros_like(y)
        
        for j in range(p):
            s = filter_step(y, 1)
            x += s
            y -= s
    return x

