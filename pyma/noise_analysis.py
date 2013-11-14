import numpy as np
import tables
import scipy.linalg as la
import pyma.noise_filter as nf

with tables.open_file("/media/data/Uni/ma/data/base_data.h5") as t:
    _data = np.array(t.get_node("/input/eta_sym"))

def frob_norm(A):
    return np.sqrt(np.sum(A ** 2))

def spec_norm(A):
    return la.svdvals(A)[0]

def get_noise(limit, entry):
    m = len(entry)
    return np.repeat(entry[limit:], m / (m - limit))[:m]

def plot_noise_svdvals(ax, limit, n, m, sample=0, **kwargs):
    noise = get_noise(limit, _data[sample,:,n,m])
    ax.plot(la.svdvals(nf.get_hankel_matrix(noise)), **kwargs)

def plot_signal_svdvals(ax, n, m, sample=0, **kwargs):
    ax.plot(la.svdvals(nf.get_hankel_matrix(_data[sample,:,n,m])), **kwargs)

def plot_noise_histogram(ax, limit, input_key="/input/eta_sym",
        norm="frob_norm", compare=True, **kwargs):
    if isinstance(norm, str):
        norm = globals()[norm]

    results = {}

    for n in range(6):
        for m in range(n+1):
            res = []
            for d in range(1000):
                entry = _data[d,:,n,m]
                noise = get_noise(limit, entry)
                N = nf.get_hankel_matrix(noise)
                H = nf.get_hankel_matrix(entry)
                value = norm(N * (N - H))
                if compare:
                    value = norm(H) / value
                res.append(value)

            results[(n,m)] = res

    hist_args = dict(histtype="step", bins=100,
            label="Noise from $t\\geq %s$" % limit, normed=False)
    hist_args.update(kwargs)

    ax.hist(np.concatenate(list(results.values())), **hist_args)
    return results

