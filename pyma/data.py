import pandas as pd
import tables
import numpy as np
import math
import itertools
import scipy.linalg as la
import scipy as sp
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

def calculate(data, algorithm):
    def gen():
        for n, d in enumerate(data):
            try:
                yield calculate_gevp(d, algorithm)
            except:
                print("%d failed" % n)

    gevs = pd.concat(gen(), keys=itertools.count())
    return gevs

def symmetrised(m):
    off_diagonal = (la.tril(m, k=-1) + la.triu(m, k=1).transpose()) / 2
    return off_diagonal + sp.diagflat(sp.diagonal(m)) + off_diagonal.transpose()

def symmetrise_data(data):
    new_one = np.zeros_like(data)
    for i, d in enumerate(new_one):
        for j, m in enumerate(d):
            new_one[i][j] = symmetrised(data[i][j])
    return new_one
