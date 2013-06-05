# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import pandas as pd
import numpy as np
import tarfile
import os
import os.path as op
import itertools

class LazyList(object):
    class __Undefined(object):
        pass

    def __init__(self, func, length):
        self._func = func
        self._cache = [self.__Undefined()] * length

    def __getitem__(self, index):
        if type(self._cache[index]) is self.__Undefined:
            self._cache[index] = self._func(index)
        return self._cache[index]

    def __len__(self):
        return len(self._cache)

class PseudoTarInfo(object):
    def __init__(self, path, name):
        self.name = name
        self.path = path

class PseudoTarfile(object):
    def __init__(self, name, simple=True):
        self._base = name
        self._members = None
        self._simple = simple

    def getmembers(self):
        if not self._members and not self._simple:
            self._members = list(
                    itertools.chain.from_iterable(
                        (PseudoTarInfo(path, name) for name in filepaths)
                        for path, _, filepaths in os.walk(self._base)
                    )
                )
        elif not self._members and self._simple:
            self._members = [
                    PseudoTarInfo(self._base, name)
                    for name in os.listdir(self._base)
                ]

        return self._members
    
    def extractfile(self, info):
        return open(op.join(info.path, info.name))

def open_tar_object(name, simple=False):
    if op.isdir(name):
        return PseudoTarfile(name, simple)
    else:
        return tarfile.open(name)


def read_file(f):
    df = pd.read_csv(f,
                     delim_whitespace=True,
                     skiprows=1,
                     names=[
                         "spin",
                         "local/smeared",
                         "time",
                         "data",
                         "data_2"
                         ]
                     )
    
    df.index = df["time"]
    del df["time"]
    del df["data_2"]
    return df

def get_matrices(tf, run, size=3):
    if type(tf) is str:
        open_tar_object(name)

    m = pd.DataFrame(
            [
                tuple(
                    map(int, i.name.split(".")[3:6])
                ) + (i,)
                for i in tf.getmembers()
            ], columns=["x", "y", "run", "info"])
    
    df = m[m.run == run]
    df["data"] = df["info"].map(lambda x: read_file(tf.extractfile(x)))

    idx = next(df["data"].items())[1].index
    max_t = idx.max().item()
    
    def get_matrix(t):
        result = np.zeros((size * 2, size * 2))
        
        for row in range(size):
            for col in range(size):
                data = df[(df.x == row) & (df.y == col)]["data"].item()

                get = lambda i: data[data["local/smeared"] == i]["data"][t + 1]

                sub_matrix = np.matrix([[get(1), get(3)], [get(5), get(7)]])

                result[2*row:2*(row+1), 2*col:2*(col+1)] = sub_matrix
            
        return np.matrix(result)

    return LazyList(get_matrix, max_t)

