# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import tarfile

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
        tf = tarfile.open(tf)

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

                sub_matrix = np.array([[get(1), get(3)], [get(5), get(7)]])

                result[2*row:2*(row+1), 2*col:2*(col+1)] = sub_matrix
            
        return result

    return LazyList(get_matrix, max_t)

