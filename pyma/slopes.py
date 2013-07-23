import numpy as np
import pandas as pd

def convolve(df, *kernels, mode="same", normalise=True):
    for k in kernels:
        if normalise:
            k = np.array(k)
            s = k.sum()
            if s:
                k = k / s

        df = df.apply(lambda x: np.convolve(x, k, mode))
    return df

def get_first_consec(array, eps=0.05):
    start = len(array)
    for n, i in enumerate(np.abs(array) < eps):
        if i:
            if n < start:
                start = n
        else:
            if n > start:
                return slice(start, n, 1)

def get_slope_simple(data, error=None, eps=0.05):
    if error is None:
        error = pd.Series(1, index=data.index)

    mean = np.log(data)
    first_deriv = np.convolve(mean, [1, -1])
    second_deriv = np.convolve(mean, [-1, 2, -1])

    s = first_deriv[get_first_consec(second_deriv, eps)]
    return s.mean(), s.std()

# def get_slope_as_exp

from scipy.optimize import curve_fit, leastsq

def _func(p, x):
    return p[0] * np.exp(-p[1] * x)

def _d_func(p, x, *args):
    v = np.exp(-p[1] * x)
    return [v, -p[0] * x * v]

def get_slope_exp_fit_2(data, error=None, range=slice(1,7)):
    res = lambda p, x, y: _func(p, x) - y
    
    if error is not None:
        res1 = lambda p, x, y: (_func(p, x) - y) / error[range].data
    else:
        res1 = res

    def residuals(p, x, y):
        if p[1] < 0:
            return 1e6
        else:
            return res1(p, x, y)

    args = (data[range].index.data, data[range].data)
    p0 = [0.4, 0.4]
    
    (popt, pcov, infodict, errmsg, ier) = \
        leastsq(residuals, p0,
            args=args,
            Dfun=_d_func,
            col_deriv=1,
            full_output=1
            )

    chi2 = np.sum(res1(popt, *args) ** 2) / (len(args[1]) - len(p0))
    pcov *= chi2

    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal solution not found: %s" % errmsg)

    return popt, pcov, chi2

def get_slope_exp_fit(data, error=None, range=slice(1,7)):
    error = pd.Series(1, index=data.index) if error is None else error

    args = (data[range].index.data, data[range].data)

    popt, pcov = curve_fit(
            lambda x, A, B: _func([A, B], x),
            args[0],
            args[1],
            np.array([0.4, 0.4]),
            sigma = error[range].data
        )

    chi2 = np.sum((_func(popt, args[0]) - args[1]) ** 2) / (len(args[1]) -
        len(popt))

    return popt, pcov, chi2
