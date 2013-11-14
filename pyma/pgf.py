import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import pandas as pd
from pyma.masses import calc_mass_curve

functions = """
masses.plot_mass
masses.plot_err
masses.plot_fit
noise_analysis.plot_noise_histogram
noise_analysis.plot_noise_svdvals
noise_analysis.plot_signal_svdvals
""".split("\n")

from functools import partial
from importlib import import_module
def _update_functions(obj):
    for name in functions:
        mod, sep, name = name.partition(".")
        if sep == "":
            continue

        func = getattr(import_module(".".join(["pyma", mod])), name)
        setattr(obj, name, partial(func, obj)) 


class PgfPlotter(object):
    def __init__(self, data_file):
        self._fig = None
        self._ax = None
        self._store = pd.HDFStore(data_file, "r")
        self._cache = dict()
        mpl.use("pgf", warn=False)

    @property
    def ax(self):
        return self._ax

    def get_mass(self, name, t_0, ev):
        key = (name, t_0, ev)
        if key not in self._cache:
            self._cache[key] = calc_mass_curve(self.store[name], t_0, ev)
        return self._cache[key]

    @property
    def store(self):
        return self._store

    def start(self, figsize, count=1):
        x, y = figsize
        figsize = count * x, y
        self._fig, (self._ax,) = plt.subplots(1, count, squeeze=False,
                                              figsize=figsize)

        for ax in self._ax:
            _update_functions(ax)
            ax._displace = 0
            ax.grid(True)
            ax.set_xlabel("$t$")
            ax.set_ylabel("$W$")
            ax.set_xlim( (0, 15) )
            ax.set_ylim( (0, 1.3) )

    def end(self):
        for ax in self._ax:
            ax.legend(fontsize=8, numpoints=1)
        self._fig.savefig(sys.stdout, format="pgf")

