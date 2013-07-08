import numpy as np
import numpy.linalg as la
import pandas as pd

import operator
def max_index(iterable):
    return max(enumerate(iterable), key=operator.itemgetter(1))

def solve_gevp(B, A=None):
    """Solve the generalised eigenvalue problem using the Cholesky
    decomposition."""
    L = np.matrix(la.cholesky(B))
    L_inv = la.inv(L)
    L_inv_t = L_inv.transpose()

    def solve_gevp(A):
        A_prime = L_inv * np.matrix(A) * L_inv_t
        return la.eigh(A_prime)

    if A is None:
        return solve_gevp
    else:
        return solve_gevp(A)

def permutation_indices(data):
     return sorted(range(len(data)), key = data.__getitem__)

# TODO: Compression
def solve_gevp_gen(a, t_0, compress=None):
    """Generator that returns the eigenvalues for t_0 -> t
       where t is in (t_0, t_max]."""
    try:
        f = solve_gevp(a[t_0])
    except la.LinAlgError:
        return

    eigenvectors = None
    count = 0
    
    for j in range(t_0 + 1, 32):
        try:
            eigenvalues, new_eigenvectors = f(a[j])
            
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)

            if j < 15:
                # TODO Sortieren nach Eigenwert
                perm = permutation_indices(eigenvalues)
            else:
                dot_products = [np.dot(e, eigenvectors / count) for e in
                        new_eigenvectors.transpose()]

                perm = [m.argmax() for m in dot_products]

            eigenvectors = eigenvectors + new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[:,perm]
                
            count += 1

            yield eigenvalues, eigenvectors / count

        except la.LinAlgError:
            pass

        
def calculate_gevp(m, compress=(lambda A, B: (A, B))):
    res_values = {}
    for i in range(32):
        ev = []
        for eigenvalues, _eigenvectors in solve_gevp_gen(m, i, compress):
            ev.append(eigenvalues)

        if len(ev):
            res_values[i] = pd.DataFrame(ev)

    return pd.concat(res_values)
