from operator import itemgetter
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd

def permutation_indices(data):
     return list(reversed(sorted(range(len(data)), key = data.__getitem__)))

def reorder_by_ev(ev1, ev2, B):
    # TODO: res seems to be broken here, investigate!
    ev1_b = np.dot(np.array(B), ev1)

    dot_products = [
            np.abs(np.dot(e, ev1_b)) for e in ev2.transpose()
            ]

    res = []
    for m in dot_products:
        for candidate in permutation_indices(m):
            if not candidate in res:
                res.append(candidate)
                break

    return res

def max_index(iterable):
    return max(enumerate(iterable), key=itemgetter(1))

def solve_gevp_gen(a, t_0, algorithm, sort_by_vectors=15, **kwargs):
    """Generator that returns the eigenvalues for t_0 -> t
       where t is in (t_0, t_max]."""
    B = np.matrix(a[t_0])
    try:
        f = algorithm(B=B, **kwargs)
    except TypeError:
        # If the function doesn't do currying, implement that here
        f = lambda A: algorithm(B=B, A=A)
    except LinAlgError:
        return

    eigenvectors = None
    count = 0

    for j in range(t_0 + 1, 32):
        try:
            eigenvalues, new_eigenvectors = f(np.matrix(a[j]))
            
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)

            if j < sort_by_vectors:
                # TODO Sortieren nach Eigenwert
                perm = permutation_indices(eigenvalues)
            else:
                perm = reorder_by_ev(new_eigenvectors, eigenvectors, B)

            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[:,perm]
                
            count += 1

            yield eigenvalues, eigenvectors

        except (LinAlgError, TypeError) as e:
            #import traceback
            #traceback.print_exc()
            pass

def calculate_gevp(m, algorithm, sort_by_vectors=99, **kwargs):
    res_values = {}
    for i in range(32):
        ev = []
        for eigenvalues, _eigenvectors in \
                solve_gevp_gen(m, i, algorithm,
                               sort_by_vectors=sort_by_vectors, **kwargs):
            ev.append(eigenvalues)

        if len(ev):
            res_values[i] = pd.DataFrame(ev)

    return pd.concat(res_values)

