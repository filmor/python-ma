from .simple_algorithms import scipy_solver
import scipy.linalg as la

def _pe_single(A):
    U, S, V_t = la.svd(A)
    _, _, V_U_t = la.svd(U)
    _, _, V_V_t = la.svd(V_t.transpose())
    return V_U_t * S * V_V_t.transpose()

def pro_esprit(B, A):
    B_ = _pe_single(B)
    A_ = _pe_single(A)
    return tuple(i.real for i in la.eig(b=B, a=A))

