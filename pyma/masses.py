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
def plot_fit(ax, curve, limits, label, color=None):
    y, yerr = curve
    full_x = y.dropna().index
    x = full_x[limits[0]:limits[1]]
    y = y[x]
    yerr = yerr[x]

    f = lambda x,a,b,c: a * np.exp(-b * x) + c

    p, pdiv = curve_fit(f, np.array(x), np.array(y), p0=(1, 1, min(y)),
            maxfev=10000, sigma=np.array(yerr**2))

    #sys.stderr.write(str(p))

#    p_err, p_errdiv = curve_fit(f, np.array(yerr.index), np.array(yerr),
#            p0=(1,-1,1))

    full_x = np.linspace(full_x[0] - 1, full_x[-1] + 1)

    ax.plot(full_x, f(full_x, *p), color=color, label=label)
    # p, p_err
    
    sum_of_weights = (1 / yerr ** 2).sum()
    # avg = (y / yerr ** 2).sum() / sum_of_weights
    # std = (((y - avg) / yerr)**2).sum() / sum_of_weights
    avg = p[2]
    std = np.sqrt(pdiv[2,2])
    
    import sys
    precision = str(int(math.ceil(-math.log10(std))) + 1)
    print(("'Fit': %." + precision + "f +- %." + precision + "f for %s") % (avg,
        std, label), file=sys.stderr)

