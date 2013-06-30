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

from scipy.optimize import curve_fit

def get_slope_exp_fit(data, error=None):
    error = pd.Series(1, index=data.index) if error is None else error

    fit_result, cov_matrix = curve_fit(
            lambda x, A, B: A * np.exp(-B * x),
            data.index.data,
            data.data,
            np.array([1., 1.])
        )

    return fit_result
