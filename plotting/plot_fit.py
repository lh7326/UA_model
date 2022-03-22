import matplotlib.pyplot as plt


def plot_ff_fit(ts, ffs, errors, f, pars):
    fig, ax = plt.subplots()
    ax.set_title('Form Factor Fit')
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('FF [GeV^-1]')
    ax.errorbar(ts, ffs, yerr=errors, ecolor='black', color='black', fmt='x')

    fit_ffs = f(ts, *pars)
    ax.scatter(ts, fit_ffs, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
