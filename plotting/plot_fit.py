import os.path
import matplotlib.pyplot as plt


def plot_ff_fit(ts, ffs, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('FF [GeV^-1]')
    ax.errorbar(ts, ffs, yerr=errors, ecolor='black', color='black', fmt='x')

    fit_ffs = f(ts, *pars)
    ax.scatter(ts, fit_ffs, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()


def plot_cs_fit(ts, ffs, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('Cross-Section [GeV^-2]')
    ax.errorbar(ts, ffs, yerr=errors, ecolor='black', color='black', fmt='x')

    fit_ffs = f(ts, *pars)
    ax.scatter(ts, fit_ffs, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()
