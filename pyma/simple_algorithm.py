import numpy as np
import numpy.linalg as la
import pandas as pd

def gevp(B):
    """Solve the generalised eigenvalue problem using the Cholesky
    decomposition."""
    L = la.cholesky(B)
    L_inv = la.inv(L)
    L_inv_t = L_inv.transpose()
    def gevp(A):
        A_prime = L_inv * A * L_inv_t
        return la.eigh(A_prime)
    return gevp

def gevp_gen(a, t_0):
    """Generator that returns the eigenvalues for t_0 -> t
       where t is in (t_0, t_max]."""
    try:
        f = gevp(a[t_0])
    except la.LinAlgError:
        return
    
    for j in range(t_0 + 1, 32):
        try:
            eigenvalues, eigenvectors = f(a[j])
            # TODO: Use the eigenvectors (index 1) to match the eigenvalues
            yield eigenvalues
        except la.LinAlgError:
            pass
        
def calculate_gevp(m):
    return pd.concat({i: pd.DataFrame(list(gevp_gen(m, i))) for i in range(32)})
