import numpy as np
import scipy.linalg as la

def simple_fix_heiberger(A, B, eps=1e-12):
    # TODO: Fehler relativ zu V(0)
    V, E = la.eigh(B)
    
    # Set all elements of the diagonal less than eps to 0 (note that we intentionally also reset all negative values)
    not_empty = V >= eps
    # Additionally split of all elements >5
    not_empty[4:] = False
    
    D = np.matrix(np.diag(V[not_empty]))
    Q = np.matrix(E[not_empty])
    
    A_0 = Q * A * Q.T
    B_0 = Q * B * Q.T
    
    R = np.matrix(np.diag(np.diag(D) ** (-0.5)))
    A_1 = R.T * A_0 * R
    B_1 = R.T * B_0 * R
    
    try:
        return la.eigh(A_1)
    except:
        pass

