import pandas as pd
import tables
import numpy as np
import math
import itertools
try:
    from gevp import calculate_gevp
except ImportError:
    from .gevp import calculate_gevp

def load_data(filename, key):
    with tables.open_file(filename) as t:
        data = np.array(t.get_node(key))
    return data

def write_data(filename, key, data):
    with tables.open_file(filename, mode="a", title="Output data") as table:
        try:
            table.remove_node(key)
        except tables.NoSuchNodeError:
            pass
        base, _, leaf = key.rpartition("/")
        table.create_array(base, leaf, data, createparents=True)

def estimate_pencil_error(data):
    # Calculate the variance of each matrix by calculating the respective
    # subvariances (i.e. $\sigma^2$) and then summing for each.
    data_variance = data[1:].var(axis=0).sum(axis=(1, 2))

    # Estimate the error of each matrix pencil using the errors (in norm) of the
    # involved matrices.
    matrix_errors = [np.sqrt(data_variance[i] + data_variance[i+1:]) for i in
            range(len(data_variance))]

    return matrix_errors

def mass_plot(t_0, ev, ax, masses):
    yerr = masses.ix[1:].std(level=1)
    y = masses.ix[0]
    
    x = y.index
        
    ax.errorbar(x, y, fmt=".", yerr=yerr)
    
    sum_of_weights = (1 / yerr ** 2).sum()
    avg = (y / yerr ** 2).sum() / sum_of_weights
    std = (((y - avg) / yerr)**2).sum() / sum_of_weights
    ax.plot(pd.Series(avg, x))
    ax.plot(pd.Series(avg + std, x))
    ax.plot(pd.Series(avg - std, x))
    
    precision = str(int(ceil(-math.log10(std))))
    print("Eigenvalue:\t%s,\tt_0:\t%s" % (ev, t_0))
    print(("'Fit': %." + precision + "f +- %." + precision + "f") % (avg, std))


def calculate(data, algorithm, filter=lambda x: x):
    gevs = pd.concat((calculate_gevp(filter(d), algorithm) for d in data),
            keys=itertools.count())
    return gevs


