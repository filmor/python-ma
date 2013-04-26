#!/bin/sh
pushd ~/dev/ma/notebooks
LD_LIBRARY_PATH=/usr/lib/llvm ipython3 notebook --pylab inline
popd
