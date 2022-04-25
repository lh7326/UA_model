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


def plot_cs_fit(ts, css, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('Cross-Section [GeV^-2]')
    ax.errorbar(ts, css, yerr=errors, ecolor='black', color='black', fmt='x')

    fit_css = f(ts, *pars)
    ax.scatter(ts, fit_css, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()


def plot_cs_fit_neutral_plus_charged(
        ts, css, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fit_css = f(ts, *pars)

    charged_ts = []
    charged_css = []
    charged_errors = []
    charged_fit = []
    neutral_ts = []
    neutral_css = []
    neutral_errors = []
    neutral_fit = []
    for i in range(len(ts)):
        datapoint = ts[i]
        if datapoint.is_charged:
            charged_ts.append(datapoint.t)
            charged_css.append(css[i])
            charged_errors.append(errors[i])
            charged_fit.append(fit_css[i])
        else:
            neutral_ts.append(datapoint.t)
            neutral_css.append(css[i])
            neutral_errors.append(errors[i])
            neutral_fit.append(fit_css[i])

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title(f'{title}: Charged')
    ax1.set_xlabel('t [GeV^2]')
    ax1.set_ylabel('Cross-Section [GeV^-2]')
    ax1.errorbar(charged_ts, charged_css, yerr=charged_errors, ecolor='black', color='black', fmt='x')
    ax1.scatter(charged_ts, charged_fit, color='red')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.set_title(f'{title}: Neutral')
    ax2.set_xlabel('t [GeV^2]')
    ax2.set_ylabel('Cross-Section [GeV^-2]')
    ax2.errorbar(neutral_ts, neutral_css, yerr=neutral_errors, ecolor='black', color='black', fmt='x')
    ax2.scatter(neutral_ts, neutral_fit, color='red')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()
