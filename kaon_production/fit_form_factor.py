from configparser import ConfigParser
import numpy as np
from scipy.optimize import curve_fit

from kaon_production.function import function_form_factor
from kaon_production.data import read_form_factor_data
from plotting.plot_fit import plot_ff_fit


def _get_bounds(t_0_isoscalar, t_0_isovector):
    """
    There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
    must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
    The masses and decay rates must be non-negative.

    """
    return (
        [t_0_isoscalar,  t_0_isovector, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
         -np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        np.inf
    )


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass)**2
    t_0_isovector = (2 * pion_mass)**2

    def partial_f(ts, *args, **kwargs):
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector, *args, **kwargs)

    ts, form_factors_values, errors = read_form_factor_data()

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
        1.570,  # mass_rho_double_prime
        0.144,  # decay_rate_rho_double_prime
        1.720,  # mass_rho_triple_prime
        0.25,  # decay_rate_rho_triple_prime
    ]

    popt, pcov = curve_fit(
        f=partial_f,
        xdata=ts,
        ydata=form_factors_values,
        p0=initial_parameters,
        sigma=None,  # errors,
        absolute_sigma=False,
        bounds=_get_bounds(t_0_isoscalar, t_0_isovector),
    )

    print(popt)

    plot_ff_fit(ts, form_factors_values, errors, partial_f, initial_parameters)
