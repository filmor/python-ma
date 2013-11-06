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


from scipy.optimize import curve_fit

def mass_plot(data, t_0, ev, ax, label, calc_masses=calc_masses_cosh):
    masses = data.xs(t_0, level=1)[ev].groupby(level=0, group_keys=False)\
            .apply(calc_masses)
    
    yerr = masses.ix[1:].std(level=1)
    y = masses.ix[0]
    
    x = y.index
        
    plot = ax.errorbar(x, y, fmt=".", yerr=yerr, label=label)
    #color = plot.lines[0].get_color()

    #f = lambda x,a,b,c: a * np.exp(-b * x) + c

    #p, pdiv = curve_fit(f, np.array(x), np.array(y), p0=(1, 1, min(y)),
    #                    sigma=np.array(1/yerr**2),
    #                    maxfev=10000)

    #sys.stderr.write(str(p))

    #p_err, p_errdiv = curve_fit(f, np.array(yerr.index),
    #        np.array(yerr), p0=(1,-1,1))

    #x = np.linspace(x[0] - 1, x[-1] + 1)
    #ax.plot(x, f(x, *p), color=color)
    # p, p_err
    
    #sum_of_weights = (1 / yerr ** 2).sum()
    #avg = (y / yerr ** 2).sum() / sum_of_weights
    # std = (((y - avg) / yerr)**2).sum() / sum_of_weights
    
    #precision = str(int(ceil(-math.log10(std))))
    #print(("'Fit': %." + precision + "f +- %." + precision + "f for %s") % (avg, std, label))

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import pandas as pd

class PgfPlotter(object):
    def __init__(self, data_file):
        self._fig = None
        self._ax = None
        self._store = pd.HDFStore(data_file, "r")
        mpl.use("pgf", warn=False)

    def start(self, figsize=(5, 4)):
        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.gca()

    def plot(self, data_set, t_0, ev, label):
        data = self._store[data_set]
        mass_plot(data, t_0, ev, self._ax, label)

    def end(self):
        self._fig.savefig(sys.stdout, format="pgf")

