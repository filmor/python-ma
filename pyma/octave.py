import oct2py
oc = oct2py.Oct2Py()
oc.addpath("../external")
oc.addpath("external")

def eigifp(A, B):
    return (-oc.eigifp(-A, B, 2), 0)

def regeig(A, B):
    ev, ew = oc.call("regeig", -A, B, nout=2)
    return -ev.transpose()[0], ew

