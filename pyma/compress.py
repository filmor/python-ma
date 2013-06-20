from numpy.linalg import svd
from numpy import diag

def direct(A, B):
    U_1, S_1, V_1_t = svd(A)
    U_2, S_2, V_2_t = svd(B)

    return U_2.transpose() * U_1 * diag(S_1) * V_1_t * V_2_t.transpose(), diag(S_2)
    
