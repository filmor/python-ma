import oct2py
import numpy as np
oc = oct2py.Oct2Py()
oc.addpath("../external")
oc.addpath("external")

def eigifp(A, B):
    try:
        res = -oc.eigifp(-A, B, 2)
        return (np.array([i for b in res for i in b]), A)
    except:
        raise TypeError

def bleigifp(A, B):
    try:
        res = -oc.bleigifp(-A, B, 2)
        return (np.array([i for b in res for i in b]), A)
    except:
        raise TypeError

def regeig(A, B):
    try:
        ev, ew = oc.call("regeig", -A, B, nout=2)
        return -ev.transpose()[0], ew
    except:
        raise TypeError

