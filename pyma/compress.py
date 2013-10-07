from numpy.linalg import svd
from numpy import diag

def direct(A, B):
    U_1, S_1, V_1_t = svd(A)
    U_2, S_2, V_2_t = svd(B)

    return U_2.transpose() * U_1 * diag(S_1) * V_1_t * V_2_t.transpose(), diag(S_2)
    
def truncate(a, n):
    z = np.zeros_like(a)
    z[:n] = a[:n]
    return z

def truncated_svd(m, n):
    u, s, v_t = la.svd(np.matrix(m))
    return np.dot(np.dot(u[:n], np.diagflat(truncate(s, n))), v_t[:,:n])

def truncated_algo(algorithm):
    def algo(B, A):
        return algorithm(truncated_svd(B, 4), truncated_svd(A, 4))
    return algo
