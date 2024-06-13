from configparser import ConfigParser
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from kaon_production.data import (
    read_data_files_new, merge_statistical_and_systematic_errors, make_function_to_remove_fsr_effects,
    KaonDatapoint)
from common.utils import make_partial_ff_or_cs_for_parameters
from model_parameters import KaonParametersSimplified
from plotting.plot_fit import plot_combined_fit


def plot_data(xss: List[List[float]], yss: List[List[float]],
              errorss: List[List[float]], labels: List[str], xlabel: str, ylabel: str, title: str,
              ylog=False, xlog=False, only_peak=False, f=None, charged=True, filepath=None,
              s_min=None, s_max=None, cross_section=True):

    assert (s_min is None and s_max is None) or not only_peak
    if only_peak:
        s_min = 1.025
        s_max = 1.055
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    formats = ['ok', '^b', 'vg', 'sr', '8c', 'py', '*m']
    assert len(formats) >= len(xss)
    formats = formats[:len(xss)]
    estimates = []
    for xs, ys, errors, legend, fmt in zip(xss, yss, errorss, labels, formats):
        if s_min is not None or s_max is not None:
            if s_max is None:
                filtered = list(filter(lambda t: s_min < t[0], zip(xs, ys, errors)))
            elif s_min is None:
                filtered = list(filter(lambda t: t[0] < s_max, zip(xs, ys, errors)))
            else:
                filtered = list(filter(lambda t: s_min < t[0] < s_max, zip(xs, ys, errors)))
            if not filtered:
                continue
            xs, ys, errors = zip(*filtered)
        ax.errorbar(xs, ys, yerr=errors, fmt=fmt, elinewidth=2, markersize=3.5, label=legend)
        if f:
            new_xs = list(xs)
            for x_low, x_high in zip(xs[:-1],xs[1:]):
                new_xs.extend([x_low + 0.01 * i * (x_high - x_low) for i in range(1, 100)])

            estimates.extend(zip(new_xs, f([[x, charged, cross_section] for x in new_xs])))

    ax.legend(loc='upper right', prop={'size': 14})
    if estimates:
        estimates = sorted(estimates, key=lambda est: est[0])
        all_xs, all_fit_vals = zip(*estimates)
        ax.plot(all_xs, all_fit_vals, '-k')

    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if filepath:
        plt.savefig(filepath, format='pdf')
    plt.show()
    plt.close()


def _prepare_data(
        ts_cs_charged: Optional[List[float]] = None, css_charged: Optional[List[float]] = None,
        cs_errors_charged: Optional[List[float]] = None,
        ts_cs_neutral: Optional[List[float]] = None, css_neutral: Optional[List[float]] = None,
        cs_errors_neutral: Optional[List[float]] = None,
        ts_ff_charged: Optional[List[float]] = None, ffs_charged: Optional[List[float]] = None,
        ff_errors_charged: Optional[List[float]] = None,
        ts_ff_neutral: Optional[List[float]] = None, ffs_neutral: Optional[List[float]] = None,
        ff_errors_neutral: Optional[List[float]] = None,
) -> Tuple[List[KaonDatapoint], List[float], List[float]]:
    ts, ys, errors = [], [], []
    if ts_cs_charged is not None:
        assert len(ts_cs_charged) == len(css_charged or []) == len(cs_errors_charged or [])
        ts += [KaonDatapoint(t, True, True) for t in ts_cs_charged]
        ys += list(css_charged or [])
        errors += list(cs_errors_charged or [])
    if ts_cs_neutral is not None:
        assert len(ts_cs_neutral) == len(css_neutral or []) == len(cs_errors_neutral or [])
        ts += [KaonDatapoint(t, False, True) for t in ts_cs_neutral]
        ys += list(css_neutral or [])
        errors += list(cs_errors_neutral or [])
    if ts_ff_charged is not None:
        assert len(ts_ff_charged) == len(ffs_charged or []) == len(ff_errors_charged or [])
        ts += [KaonDatapoint(t, True, False) for t in ts_ff_charged]
        ys += list(ffs_charged or [])
        errors += list(ff_errors_charged or [])
    if ts_ff_neutral is not None:
        assert len(ts_ff_neutral) == len(ffs_neutral or []) == len(ff_errors_neutral or [])
        ts += [KaonDatapoint(t, False, False) for t in ts_ff_neutral]
        ys += list(ffs_neutral or [])
        errors += list(ff_errors_neutral or [])

    ts, ys, errors = zip(
        *sorted(
            zip(ts, ys, errors),
            key=lambda tup: tup[0].t,
        )
    )
    return ts, ys, errors


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports/kaons'

    remove_fsr_effects = make_function_to_remove_fsr_effects(charged_kaon_mass, alpha)

    def discard_above_threshold(threshold, xs, ys, ers):
        return xs, ys, ers
        # return list(zip(*filter(lambda t: t[0] < threshold, zip(xs, ys, ers))))

    THRESHOLD = 10  # GeV^2

    (timelike_charged_ts, timelike_charged_cross_sections_values,
     timelike_charged_errors) = discard_above_threshold(THRESHOLD, *remove_fsr_effects(
        *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    'cmd_3_charged_kaons_undressed.csv',
                    'babar_2013_charged_kaons_undressed.csv',
                    'babar_charged_kaons_2015_undressed.csv',
                    'BESIII_charged_kaons_2019_undressed.csv',
                ]
            )
        )
    ))
    (timelike_neutral_ts, timelike_neutral_cross_sections_values,
     timelike_neutral_errors) = discard_above_threshold(THRESHOLD, *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    'cmd_2_neutral_kaons_undressed.csv',  # added
                    'cmd_3_neutral_kaons_undressed.csv',
                    'babar_neutral_kaons_2014_undressed.csv',
                    'BESIII_neutral_kaons_2021_undressed.csv',
                ]
            )
    ))

    (spacelike_charged_ts, spacelike_charged_form_factor_values,
     space_charged_errors) = merge_statistical_and_systematic_errors(
        *read_data_files_new(
            file_names=[
                'spacelike_charged_kaons_formfactor_1980_undressed.csv',
                'spacelike_charged_kaons_formfactor_1986_undressed.csv',
            ]
        )
    )

    ts, ys, errs = _prepare_data(
        ts_cs_charged=timelike_charged_ts, css_charged=timelike_charged_cross_sections_values,
        cs_errors_charged=timelike_charged_errors,
        ts_cs_neutral=timelike_neutral_ts, css_neutral=timelike_neutral_cross_sections_values,
        cs_errors_neutral=timelike_neutral_errors,
        ts_ff_charged=spacelike_charged_ts, ffs_charged=spacelike_charged_form_factor_values,
        ff_errors_charged=space_charged_errors,
    )

    #kaon_parameters_filepath = f'/home/lukas/reports/kaons/article_fit/final_fit_parameters.pickle'
    kaon_parameters_filepath = f'/home/lukas/reports/kaons/article2_fit/final_fit_parameters.pickle'
    kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
    # kaon_parameters = KaonParametersSimplified(
    #     t_0_isoscalar=0.17531904388276887,
    #     t_0_isovector=0.07791957505900839,
    #     t_in_isoscalar=0.61253717,
    #     t_in_isovector=1.77195907,
    #     a_omega=0.19772368,
    #     mass_omega=0.78266,
    #     decay_rate_omega=0.00868,
    #     a_omega_double_prime=0.15791981,
    #     mass_omega_double_prime=1.67,
    #     decay_rate_omega_double_prime=0.32915526,
    #     a_phi=0.32426641,
    #     mass_phi=1.01903667,
    #     decay_rate_phi=0.004139,
    #     a_phi_prime=-0.18332132,
    #     mass_phi_prime=1.63963787,
    #     decay_rate_phi_prime=0.23220697,
    #     mass_phi_double_prime=2.20628006,
    #     decay_rate_phi_double_prime=0.1000116,
    #     a_rho=0.53077563,
    #     mass_rho=0.75823,
    #     decay_rate_rho=0.14456,
    #     a_rho_prime=-0.13076398,
    #     mass_rho_prime=1.46675669,
    #     decay_rate_rho_prime=0.79642807,
    #     mass_rho_double_prime=1.86404824,
    #     decay_rate_rho_double_prime=0.48068968,
    # )

    free_pars = kaon_parameters.get_free_values()
    kaon_parameters.fix_all_parameters()
    f = make_partial_ff_or_cs_for_parameters(
        alpha, hc_squared, kaon_parameters,
        charged_kaon_mass=charged_kaon_mass,
        neutral_kaon_mass=neutral_kaon_mass,
    )

    fit_ys = f(ts)
    r_squared = [(data - fit) ** 2 for data, fit in zip(ys, fit_ys)]
    chi_squared_total = (
        sum([r2 / (err ** 2) for r2, err in zip(r_squared, errs)])
    ) / (len(r_squared) - len(free_pars))

    training_ts, training_ys, training_errs = zip(
        *filter(lambda tup: tup[0].is_for_cross_section,
                zip(ts, ys, errs))
    )
    fit_training_ys = f(training_ts)
    r_squared_training = [(data - fit) ** 2 for data, fit in zip(training_ys, fit_training_ys)]
    chi_squared_training_set = (
        sum([r2 / (err ** 2) for r2, err in zip(r_squared_training, training_errs)])
    ) / (len(r_squared_training) - len(free_pars))

    print(f'Chi squared total: {chi_squared_total}\nChi squared training set: {chi_squared_training_set}')
    print(kaon_parameters.to_list())

    xss, yss, errss, labels = [], [], [], []
    # for label, filename in [
    #     ('CMD3', 'cmd_3_charged_kaons_undressed.csv') ,
    #     ('BaBar2013', 'babar_2013_charged_kaons_undressed.csv'),
    #     ('BaBar2015', 'babar_charged_kaons_2015_undressed.csv'),
    #     ('BESIII', 'BESIII_charged_kaons_2019_undressed.csv'),
    #     ]:
    #     ts, css, errs = discard_above_threshold(THRESHOLD, *remove_fsr_effects(
    #         *merge_statistical_and_systematic_errors(*read_data_files_new(file_names=[filename]))))
    #     xss.append(ts)
    #     yss.append(css)
    #     errss.append(errs)
    #     labels.append(label)

    # for label, filename in [
    #     ('CMD2', 'cmd_2_neutral_kaons_undressed.csv'),
    #     ('CMD3', 'cmd_3_neutral_kaons_undressed.csv') ,
    #     ('BaBar', 'babar_neutral_kaons_2014_undressed.csv'),
    #     ('BESIII', 'BESIII_neutral_kaons_2021_undressed.csv'),
    #     ]:
    #     ts, css, errs = discard_above_threshold(THRESHOLD, *remove_fsr_effects(
    #         *merge_statistical_and_systematic_errors(*read_data_files_new(file_names=[filename]))))
    #     xss.append(ts)
    #     yss.append(css)
    #     errss.append(errs)
    #     labels.append(label)

    for label, filename in [
        ('Dally et al.', 'spacelike_charged_kaons_formfactor_1980_undressed.csv'),
        ('Amendolia et al.', 'spacelike_charged_kaons_formfactor_1986_undressed.csv'),
    ]:
        ts, css, errs = merge_statistical_and_systematic_errors(*read_data_files_new(file_names=[filename]))
        xss.append(ts)
        yss.append(css)
        errss.append(errs)
        labels.append(label)

    plot_data(xss, yss, errss, labels, 's [GeV^2]', 'Form factor', 'Charged kaons fit',
              ylog=False, only_peak=False, f=f, charged=True,
              s_min=None, s_max=None, cross_section=False,
              filepath='/home/lukas/latex_projects/clanok_kaon_model/figures/charged_kaons_fit_spacelike.pdf')
