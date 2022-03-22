import matplotlib.pyplot as plt


def plot_ff_fit(ts, ffs, errors, f, pars):
    fig, ax = plt.subplots()
    ax.set_title('Form Factor Fit')
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('FF [GeV^-1]')
    ax.errorbar(ts, ffs, yerr=errors, ecolor='black', color='black', fmt='x')

    #min_ts = min(ts)
    #max_ts = max(ts)
    #fit_ts = [min_ts + (max_ts - min_ts) * x / 100.0 for x in range(101)]
    fit_ffs = [f(t, *pars) for t in ts]
    ax.scatter(ts, fit_ffs, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
