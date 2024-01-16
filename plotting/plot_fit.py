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


def plot_combined_fit(
        ts, ys, errors, f, pars, title='Combined Fit', show=True, save_dir=None):

    def _plot(ax, title, y_label, x_vals, y_vals, error_vals, fit_vals):
        ax.set_title(title)
        ax.set_xlabel('t [GeV^2]')
        ax.set_ylabel(y_label)
        ax.errorbar(x_vals, y_vals, yerr=error_vals, ecolor='black', color='black', fmt='x')
        ax.scatter(x_vals, fit_vals, color='red')

        ax.set_yscale('log')
        if min(x_vals) > 0:
            ax.set_xscale('log')

    fit_ys = f(ts, *pars)

    charged_cs_ts = []
    charged_cs_ys = []
    charged_cs_errors = []
    charged_cs_fit = []
    neutral_cs_ts = []
    neutral_cs_ys = []
    neutral_cs_errors = []
    neutral_cs_fit = []
    charged_ff_ts = []
    charged_ff_ys = []
    charged_ff_errors = []
    charged_ff_fit = []
    neutral_ff_ts = []
    neutral_ff_ys = []
    neutral_ff_errors = []
    neutral_ff_fit = []

    for i in range(len(ts)):
        datapoint = ts[i]
        if datapoint.is_charged and datapoint.is_for_cross_section:
            charged_cs_ts.append(datapoint.t)
            charged_cs_ys.append(ys[i])
            charged_cs_errors.append(errors[i])
            charged_cs_fit.append(fit_ys[i])
        elif datapoint.is_charged and not datapoint.is_for_cross_section:
            charged_ff_ts.append(datapoint.t)
            charged_ff_ys.append(ys[i])
            charged_ff_errors.append(errors[i])
            charged_ff_fit.append(fit_ys[i])
        elif not datapoint.is_charged and datapoint.is_for_cross_section:
            neutral_cs_ts.append(datapoint.t)
            neutral_cs_ys.append(ys[i])
            neutral_cs_errors.append(errors[i])
            neutral_cs_fit.append(fit_ys[i])
        elif not datapoint.is_charged and not datapoint.is_for_cross_section:
            neutral_ff_ts.append(datapoint.t)
            neutral_ff_ys.append(ys[i])
            neutral_ff_errors.append(errors[i])
            neutral_ff_fit.append(fit_ys[i])
        else:
            raise ValueError(f'Unexpected datapoint value: {datapoint}')

    nr_subplots = 0
    if charged_cs_ts:
        nr_subplots += 1
    if charged_ff_ts:
        nr_subplots += 1
    if neutral_cs_ts:
        nr_subplots += 1
    if neutral_ff_ts:
        nr_subplots += 1

    if nr_subplots == 4:
        nr_rows = 2
        nr_cols = 2
    else:
        nr_rows = nr_subplots
        nr_cols = 1

    fig, axes = plt.subplots(nrows=nr_rows, ncols=nr_cols)
    fig.set_figwidth(12.8)
    fig.set_figheight(9.6)
    if nr_subplots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    if charged_cs_ts:
        ax1 = axes.pop()
        _plot(ax1, f'{title}: Charged CS', 'Cross Section [nb]',
              charged_cs_ts, charged_cs_ys, charged_cs_errors, charged_cs_fit)
    if charged_ff_ts:
        ax2 = axes.pop()
        _plot(ax2, f'{title}: Charged FF', 'Form Factor [1]',
              charged_ff_ts, charged_ff_ys, charged_ff_errors, charged_ff_fit)
    if neutral_cs_ts:
        ax3 = axes.pop()
        _plot(ax3, f'{title}: Neutral CS', 'Cross Section [nb]',
              neutral_cs_ts, neutral_cs_ys, neutral_cs_errors, neutral_cs_fit)
    if neutral_ff_ts:
        ax4 = axes.pop()
        _plot(ax4, f'{title}: Neutral FF', 'Form Factor [1]',
              neutral_ff_ts, neutral_ff_ys, neutral_ff_errors, neutral_ff_fit)

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
