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
                return slice(start, n)

def get_slopes(gevps, eps=0.05):
    gevps_log = np.log(gevps)

    results = []

    for run, sample in gevps_log.groupby(level=0):
        for t_0, df in sample.groupby(level=1):
            first_deriv = convolve(df, [1, -1])
            second_deriv = convolve(df, [-1, 2, -1])
            
            for i in gevps:
                try:
                    s = first_deriv[i][get_first_consec(second_deriv[i], eps)]
                    results.append((i, run, t_0, s.mean(), s.std(), s.count()))
                except KeyError:
                    continue
        
    return pd.DataFrame(results,
                        columns="eigenvalue run t_0 mean std count".split()
            ).set_index(["eigenvalue", "run", "t_0"])
