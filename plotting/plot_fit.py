import os.path
import matplotlib.pyplot as plt


def plot_ff_fit(ts, ffs, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('FF [1]')
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


def plot_cs_fit(ts, css, errors, f, pars, title='Cross Section Fit', show=True, save_dir=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('t [GeV^2]')
    ax.set_ylabel('Cross-Section [GeV^-2]')
    xs = [datapoint.t for datapoint in ts]
    ax.errorbar(xs, css, yerr=errors, ecolor='black', color='black', fmt='x')

    fit_css = f(ts, *pars)
    ax.scatter(xs, fit_css, color='red')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()


def plot_cs_fit_neutral_plus_charged(
        ts, css, errors, f, pars, title='Cross Section Fit', show=True, save_dir=None):
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


def plot_ff_fit_neutral_plus_charged(
        ts, ffs, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fit_ffs = f(ts, *pars)

    charged_ts = []
    charged_ffs = []
    charged_errors = []
    charged_fit = []
    neutral_ts = []
    neutral_ffs = []
    neutral_errors = []
    neutral_fit = []
    for i in range(len(ts)):
        datapoint = ts[i]
        if datapoint.is_charged:
            charged_ts.append(datapoint.t)
            charged_ffs.append(ffs[i])
            charged_errors.append(errors[i])
            charged_fit.append(fit_ffs[i])
        else:
            neutral_ts.append(datapoint.t)
            neutral_ffs.append(ffs[i])
            neutral_errors.append(errors[i])
            neutral_fit.append(fit_ffs[i])

    nr_subplots = 0
    if charged_ts:
        nr_subplots += 1
    if neutral_ts:
        nr_subplots += 1

    fig, axes = plt.subplots(nr_subplots, 1)
    if nr_subplots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    if charged_ts:
        ax1 = axes.pop()
        ax1.set_title(f'{title}: Charged')
        ax1.set_xlabel('t [GeV^2]')
        ax1.set_ylabel('Form Factor [1]')
        ax1.errorbar(charged_ts, charged_ffs, yerr=charged_errors, ecolor='black', color='black', fmt='x')
        ax1.scatter(charged_ts, charged_fit, color='red')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    if neutral_ts:
        ax2 = axes.pop()
        ax2.set_title(f'{title}: Neutral')
        ax2.set_xlabel('t [GeV^2]')
        ax2.set_ylabel('Form Factor [1]')
        ax2.errorbar(neutral_ts, neutral_ffs, yerr=neutral_errors, ecolor='black', color='black', fmt='x')
        ax2.scatter(neutral_ts, neutral_fit, color='red')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()


def plot_background_residuals(
        ts, ffs, background_fit, errors, ts_residuals, residuals, errors_residuals, f, pars,
        title='Background residuals', show=True, save_dir=None):
    fit_res = f(ts_residuals, *pars)
    ts = [datapoint.t for datapoint in ts]
    ts_residuals = [datapoint.t for datapoint in ts_residuals]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title(f'{title}: Effective FF')
    ax1.set_xlabel('p [GeV]')
    ax1.set_ylabel('FF [1]')
    ax1.errorbar(ts, ffs, yerr=errors, ecolor='black', color='black', fmt='x')
    ax1.scatter(ts, background_fit, color='red')

    ax2.set_title(f'{title}: Residues')
    ax2.set_xlabel('p [GeV]')
    ax2.set_ylabel('Background Residues [1]')
    ax2.errorbar(ts_residuals, residuals, yerr=errors_residuals,
                 ecolor='black', color='black', fmt='x')
    ax2.scatter(ts_residuals, fit_res, color='red')
    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()


def plot_ff_fit_electric_plus_magnetic(
        ts, ffs, errors, f, pars, title='Form Factor Fit', show=True, save_dir=None):
    fit_ffs = f(ts, *pars)

    electric_ts = []
    electric_ffs = []
    electric_errors = []
    electric_fit = []
    magnetic_ts = []
    magnetic_ffs = []
    magnetic_errors = []
    magnetic_fit = []
    for i in range(len(ts)):
        datapoint = ts[i]
        if datapoint.electric:
            electric_ts.append(datapoint.t)
            electric_ffs.append(ffs[i])
            electric_errors.append(errors[i])
            electric_fit.append(fit_ffs[i])
        else:
            magnetic_ts.append(datapoint.t)
            magnetic_ffs.append(ffs[i])
            magnetic_errors.append(errors[i])
            magnetic_fit.append(fit_ffs[i])

    nr_subplots = 0
    if electric_ts:
        nr_subplots += 1
    if magnetic_ts:
        nr_subplots += 1

    fig, axes = plt.subplots(nr_subplots, 1)
    if nr_subplots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    if electric_ts:
        ax1 = axes.pop()
        ax1.set_title(f'{title}: Electric FF')
        ax1.set_xlabel('t [GeV^2]')
        ax1.set_ylabel('Form Factor [1]')
        ax1.errorbar(electric_ts, electric_ffs, yerr=electric_errors, ecolor='black', color='black', fmt='x')
        ax1.scatter(electric_ts, electric_fit, color='red')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    if magnetic_ts:
        ax2 = axes.pop()
        ax2.set_title(f'{title}: Magnetic FF')
        ax2.set_xlabel('t [GeV^2]')
        ax2.set_ylabel('Form Factor [1]')
        ax2.errorbar(magnetic_ts, magnetic_ffs, yerr=magnetic_errors, ecolor='black', color='black', fmt='x')
        ax2.scatter(magnetic_ts, magnetic_fit, color='red')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    if show:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{title.lower()}.png'))
    plt.close()
