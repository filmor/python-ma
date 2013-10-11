
def draw_matrix(m):
    plt.figure()
    plt.gca().invert_yaxis()
    pcolor(m)

def get_median_and_errors(series):
    p = 0.683
    lower = series.quantile(1 - p)
    upper = series.quantile(p)
    return upper, series.quantile(0.5), lower, (upper - lower) / 2

from scipy.optimize import curve_fit

def get_mass(dt, dt_err):
    f = lambda x,a,b,c: a * np.exp(-b * x) + c

    p, pdiv = curve_fit(f, np.array(dt.index), np.array(dt), p0=(1, 1, 0.5),
            sigma=np.array(1/dt_err**2), maxfev=10000)
    p_err, p_errdiv = curve_fit(f, np.array(dt_err.index), np.array(dt_err),
            p0=(1,-1,1))

    x = np.linspace(dt.index[0] - 1, dt.index[-1] + 1)
    gca().plot(x, f(x, *p))
    gca().errorbar(dt.index, dt, yerr=dt_err, fmt=".")
    p, p_err
