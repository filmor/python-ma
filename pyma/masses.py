import math
import numpy as np
import pandas as pd

def _set_false_after_first(val):
    start = len(val)
    for n, i in enumerate(val):
        if i:
            if n < start:
                start = n
        else:
            if n > start:
                val[n:] = False
                break

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

def calc_mass_curve(data, t_0, ev, calc_masses=calc_masses_cosh,
        filter=lambda x: x):

    def do_calc(x):
        y = filter(x)
        return calc_masses(pd.Series(y, index=x.index))

    masses = data.xs(t_0, level=1)[ev].groupby(level=0, group_keys=False)\
            .apply(do_calc)
    return masses.ix[0].shift(t_0), \
           masses.ix[1:].std(level=1).shift(t_0)

def plot_mass(ax, curve, label):
    y, yerr = curve
    x = y.index + getattr(ax, "_displace", 0)
    select = yerr < 0.9 * y
    _set_false_after_first(select)
    plot = ax.errorbar(np.array(x[select]), np.array(y[select]), fmt=".",
            yerr=np.array(yerr[select]), label=label)
    return plot.lines[0].get_color()

def plot_err(ax, curve, label):
    y, yerr = curve
    x = y.index + getattr(ax, "_displace", 0)
    plot = ax.plot(x, np.array(y / yerr), label=label)

from scipy.optimize import curve_fit
def plot_fit(ax, curve, initial, color=None):
    x = y.index

    f = lambda x,a,b,c: a * np.exp(-b * x) + c
    p, pdiv = curve_fit(f, np.array(x), np.array(y), p0=(1, 1, min(y)),
                        sigma=np.array(1/yerr**2),
                        maxfev=10000)

    #sys.stderr.write(str(p))

    p_err, p_errdiv = curve_fit(f, np.array(yerr.index), np.array(yerr),
            p0=(1,-1,1))

    x = np.linspace(x[0] - 1, x[-1] + 1)
    ax.plot(x, f(x, *p), color=color)
    # p, p_err
    
    sum_of_weights = (1 / yerr ** 2).sum()
    avg = (y / yerr ** 2).sum() / sum_of_weights
    std = (((y - avg) / yerr)**2).sum() / sum_of_weights
    
    # precision = str(int(ceil(-math.log10(std))))
    # (("'Fit': %." + precision + "f +- %." + precision + "f for %s") % (avg, std, label))

