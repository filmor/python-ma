import oct2py
import threading
import numpy as np

oc = oct2py.Oct2Py()
n = 0

def _do_restart(*args, **kwargs):
    global n
    n += 1
    print("Restarting %s" % n)
    # print("Args: ", *args)
    _restart()
    print("Done %s" % n)

def _restart(silent=False):
    global oc
    oc.restart()
    try:
        oc.addpath("../external")
        print("Adding path ../external")
    except:
        pass
    try:
        oc.addpath("external")
        print("Adding path external")
    except:
        pass

_restart()

from time import time

def timeout(t):
    def decorator(func):
        def decorated(*args, **kwargs):
            timer = threading.Timer(t, lambda: _do_restart)
            start = time()
            timer.start()
            try:
                res = func(*args, **kwargs)
                if timer.finished.is_set():
                    print("Timer finished")
                    print(time() - start, "vs", t)
                    raise TypeError
                    #raise RuntimeError("Timeout")
                timer.cancel()
                return res
            except AttributeError:
                print("Attribute error")
                timer.cancel()
                print(time() - start, "vs", t)
                raise
                #raise RuntimeError("Timeout")
                # raise RuntimeError("Timeout")
        return decorated

    return decorator

@timeout(0.5)
def eigifp(A, B):
    try:
        res = -oc.eigifp(-A, B, 2)
        return (np.array([i for b in res for i in b]), A)
    except oct2py.Oct2PyError:
        raise TypeError

@timeout(0.5)
def bleigifp(A, B):
    try:
        res = -oc.bleigifp(-A, B, 2)
        return (np.array([i for b in res for i in b]), A)
    except:
        raise TypeError

@timeout(0.5)
def regeig(A, B):
    try:
        ev, ew = oc.call("regeig", -A, B, nout=2)
        return -ev.transpose()[0], ew
    except:
        raise TypeError

