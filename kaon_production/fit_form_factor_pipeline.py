from configparser import ConfigParser
import math
import numpy as np
from scipy.optimize import curve_fit

from kaon_production.data import read_form_factor_data
from kaon_production.utils import (
    get_bounds, get_bounds_low_energy, report_fit, make_partial_full, make_partial_low_energy, make_partial_high_energy,
    make_partial_fixed_resonances, make_partial_only_resonances_parameters)
from plotting.plot_fit import plot_ff_fit


INITIAL_PARAMETERS = [
    1.12110178,  # t_in_isoscalar
    1.49572898,  # t_in_isovector
    2.06984628,  # a_omega
    -2.19271179,  # a_omega_prime
    0.31957227,  # a_omega_double_prime
    0.25334654,  # a_phi
    0.08406246,  # a_phi_prime
    -0.58567654,  # a_rho
    1.23174638,  # a_rho_prime
    -0.1842823,  # a_rho_double_prime
    0.78266,  # mass_omega
    0.00868,  # decay_rate_omega
    1.410,  # mass_omega_prime
    0.29,  # decay_rate_omega_prime
    1.670,  # mass_omega_double_prime
    0.315,  # decay_rate_omega_double_prime
    1.019461,  # mass_phi
    0.004249,  # decay_rate_phi
    1.680,  # mass_phi_prime
    0.150,  # decay_rate_phi_prime
    2.159,  # mass_phi_double_prime
    0.137,  # decay_rate_phi_double_prime
    0.77525,  # mass_rho
    0.1474,  # decay_rate_rho
    1.465,  # mass_rho_prime
    0.4,  # decay_rate_rho_prime
    1.720,  # mass_rho_double_prime
    0.25,  # decay_rate_rho_double_prime
    2.15,  # mass_rho_triple_prime
    0.3,  # decay_rate_rho_triple_prime
]


def _split_data_into_low_and_high_energies(ts, ffs, errors, splitting_t=1.5):
    ts_low = []
    ffs_low = []
    errors_low = []
    ts_high = []
    ffs_high = []
    errors_high = []
    for t, ff, err in zip(ts, ffs, errors):
        if t < splitting_t:
            ts_low.append(t)
            ffs_low.append(ff)
            errors_low.append(err)
        else:
            ts_high.append(t)
            ffs_high.append(ff)
            errors_high.append(err)
    return ts_low, ffs_low, errors_low, ts_high, ffs_high, errors_high


def pipeline(ts, form_factors_values, errors, path_to_config):
    # let us first try this in a straightforward way

    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    ts_low, ffs_low, errors_low, ts_high, ffs_high, errors_high = _split_data_into_low_and_high_energies(
        ts, form_factors_values, errors
    )

    initial_parameters = [
        1.0,  # t_in_isoscalar
        1.0,  # t_in_isovector
        1.0 / 6,  # a_omega
        0.78266,  # mass_omega
        0.00868,  # decay_rate_omega
        1.0 / 6,  # a_phi
        1.019461,  # mass_phi
        0.004249,  # decay_rate_phi
        1.410,  # mass_omega_prime
        0.29,  # decay_rate_omega_prime
        1.0 / 4,  # a_rho
        0.77525,  # mass_rho
        0.1474,  # decay_rate_rho
        1.465,  # mass_rho_prime
        0.4,  # decay_rate_rho_prime
    ]

    partial_low_energy = make_partial_low_energy(path_to_config)

    popt, _ = curve_fit(
        f=partial_low_energy,
        xdata=ts_low,
        ydata=ffs_low,
        p0=initial_parameters,
        sigma=errors_low,
        absolute_sigma=False,
        bounds=get_bounds_low_energy(t_0_isoscalar, t_0_isovector),
    )

    print(report_fit(ts, form_factors_values, errors, popt, partial_low_energy))
    plot_ff_fit(ts, form_factors_values, errors, partial_low_energy, popt, 'Low Energy')


def pipeline2(ts, form_factors_values, errors, path_to_config):
    # let us first try this in a straightforward way

    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    partial = make_partial_fixed_resonances(
        path_to_config, 0.78266, 0.00868, 1.410, 0.29, 1.67, 0.315,
        1.019461, 0.004249, 1.680, 0.150, 2.159, 0.137,
        0.77526, 0.1474, 1.465, 0.4, 1.720, 0.25, 2.15, 0.3
    )

    initial_parameters = [
        1.0,  # t_in_isoscalar
        1.0,  # t_in_isovector
        1.0 / 12,  # a_omega
        1.0 / 12,  # a_omega_prime
        1.0 / 12,  # a_omega_double_prime
        1.0 / 12,  # a_phi
        1.0 / 12,  # a_phi_prime
        1.0 / 8,  # a_rho
        1.0 / 8,  # a_rho_prime
        1.0 / 8,  # a_rho_double_prime
    ]

    popt, _ = curve_fit(
        f=partial,
        xdata=ts,
        ydata=form_factors_values,
        p0=initial_parameters,
        sigma=errors,
        absolute_sigma=False,
        bounds=(get_bounds(t_0_isoscalar, t_0_isovector)[0][:10], np.inf),
        maxfev=5000,
    )

    print(report_fit(ts, form_factors_values, errors, popt, partial))
    plot_ff_fit(ts, form_factors_values, errors, partial, popt, 'Fixed resonances')


def pipeline3(ts, form_factors_values, errors, path_to_config):
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    partial = make_partial_only_resonances_parameters(
        path_to_config, 1.12110178,  1.49572898,  2.06984628, -2.19271179,  0.31957227,
        0.25334654,  0.08406246, -0.58567654,  1.23174638, -0.1842823
    )

    initial_parameters = [
        0.78266, 0.00868, 1.410, 0.29, 1.67, 0.315,
        1.019461, 0.004249, 1.680, 0.150, 2.159, 0.137,
        0.77526, 0.1474, 1.465, 0.4, 1.720, 0.25, 2.15, 0.3
    ]

    lower_mass_bound_isoscalar = math.sqrt(t_0_isoscalar)
    lower_mass_bound_isovector = math.sqrt(t_0_isovector)

    popt, _ = curve_fit(
        f=partial,
        xdata=ts,
        ydata=form_factors_values,
        p0=initial_parameters,
        sigma=errors,
        absolute_sigma=False,
        bounds=((lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0,
                 lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0,
                 lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isovector, 0.0, lower_mass_bound_isovector, 0.0,
                 lower_mass_bound_isovector, 0.0, lower_mass_bound_isovector, 0.0),
                np.inf),
        maxfev=5000,
    )

    print(report_fit(ts, form_factors_values, errors, popt, partial))
    plot_ff_fit(ts, form_factors_values, errors, partial, popt, 'Fixed resonances')


if __name__ == '__main__':
    ts, form_factors_values, errors = read_form_factor_data()

    pipeline3(ts, form_factors_values, errors, '../configuration.ini')
