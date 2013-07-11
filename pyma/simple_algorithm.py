import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import pandas as pd

def cholesky_solver(B, A=None):
    """Solve the generalised eigenvalue problem using the Cholesky
    decomposition."""
    L = la.cholesky(B)
    L_inv = la.inv(L)
    L_inv_t = L_inv.transpose()

    def solve_gevp(A):
        A_prime = L_inv * A * L_inv_t
        return la.eigh(A_prime)

    if A is None:
        return solve_gevp
    else:
        return solve_gevp(A)

def qz_solver(B, A):
    # A - \lambda B = Q(S - \lambda T)Z*
    S, T, Q, Z = sla.qz(A=A, B=B)
    eigenvalues = np.array(S).diagonal() / np.array(T).diagonal()

    # TODO: Calculate eigenvectors
    return eigenvalues, Q

