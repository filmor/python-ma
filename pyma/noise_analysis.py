import numpy as np
import tables
import scipy.linalg as la
import pyma.noise_filter as nf

def frob_norm(A):
    return np.sqrt(np.sum(A ** 2))

def spec_norm(A):
    return la.svdvals(A)[0]

def plot_noise_histogram(ax, limit, input_key="/input/eta_sym",
        norm="frob_norm", compare=True, **kwargs):
    if isinstance(norm, str):
        norm = globals()[norm]

    with tables.open_file("/media/data/Uni/ma/data/base_data.h5") as t:
        data = np.array(t.get_node("/input/eta_sym"))

    results = {}

    for n in range(6):
        for m in range(n):
            res = []
            for d in range(1000):
                entry = data[d,:,n,m]
                noise = np.repeat(entry[limit:], 32 / (32 - limit))
                N = nf.get_hankel_matrix(noise)
                H = nf.get_hankel_matrix(entry)
                value = norm(N * (N - H))
                if compare:
                    value = norm(H) / value
                res.append(value)

            results[(n,m)] = res

    hist_args = dict(histtype="step", bins=100,
            label="Noise from $t\\geq %s$" % limit, normed=True)
    hist_args.update(kwargs)

    ax.hist(np.concatenate(list(results.values())), **hist_args)
    return results

