import math
import numpy as np

def acosh(x):
    try:
        return math.acosh(x)
    except ValueError:
        return None

def calc_masses_cosh(arr):
    diff = (arr.shift(-1) + arr.shift(1)) / (2 * arr)
    return diff.apply(acosh)

def calc_masses_exp(arr):
    return np.log(arr.shift(1) / arr)

def calc_masses(gevs, method="cosh"):
    if method == "exp":
        method = calc_masses_exp
    elif method == "cosh":
        method = calc_masses_cosh
    else:
        raise ValueError("Should be cosh or exp")

    return gevs.groupby(level=[0, 1],
            group_keys=False).apply(calc_masses_cosh).reorder_levels([1,2,0])

