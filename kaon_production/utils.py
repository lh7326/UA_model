#TODO: refactor!
from configparser import ConfigParser
import math
import numpy as np

from kaon_production.function import function_form_factor


def get_bounds(t_0_isoscalar, t_0_isovector):
    """
    There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
    must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
    Decay rates must be non-negative and squared masses must lie above their respective t_0 treshold.

    """
    lower_mass_bound_isoscalar = math.sqrt(t_0_isoscalar)
    lower_mass_bound_isovector = math.sqrt(t_0_isovector)
    return (
        [t_0_isoscalar,  t_0_isovector, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
         -np.inf, -np.inf, -np.inf, lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0,
         lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0,
         lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isovector, 0.0, lower_mass_bound_isovector, 0.0,
         lower_mass_bound_isovector, 0.0, lower_mass_bound_isovector, 0.0],
        np.inf
    )


def report_fit(xs, data_ys, errors, parameters, f_partial):

    fit_ys = f_partial(xs, *parameters)
    r_squared_absolute = [(data - fit)**2 for data, fit in zip(data_ys, fit_ys)]
    r_squared_errors_normalized = [r2 / (err ** 2) for r2, err in zip(r_squared_absolute, errors)]
    return {
        'parameters': parameters,
        'r2': sum(r_squared_absolute),
        'r2_normalized': sum(r_squared_errors_normalized),
    }


def _read_config(path_to_config):
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector


def make_partial_full(path_to_config):
    t_0_isoscalar, t_0_isovector = _read_config(path_to_config)

    def partial_f(ts, *args, **kwargs):
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector, *args, **kwargs)

    return partial_f


def make_partial_fixed_resonances(
        path_to_config, mass_omega, decay_rate_omega, mass_omega_prime, decay_rate_omega_prime,
        mass_omega_double_prime, decay_rate_omega_double_prime, mass_phi, decay_rate_phi,
        mass_phi_prime, decay_rate_phi_prime, mass_phi_double_prime, decay_rate_phi_double_prime,
        mass_rho, decay_rate_rho, mass_rho_prime, decay_rate_rho_prime,
        mass_rho_double_prime, decay_rate_rho_double_prime, mass_rho_triple_prime, decay_rate_rho_triple_prime):
    t_0_isoscalar, t_0_isovector = _read_config(path_to_config)

    def partial_f(ts, t_in_isoscalar, t_in_isovector, a_omega, a_omega_prime, a_omega_double_prime,
                  a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime):
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector,
                                    t_in_isoscalar, t_in_isovector,
                                    a_omega, a_omega_prime, a_omega_double_prime,
                                    a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime,
                                    mass_omega, decay_rate_omega,
                                    mass_omega_prime, decay_rate_omega_prime,
                                    mass_omega_double_prime, decay_rate_omega_double_prime,
                                    mass_phi, decay_rate_phi,
                                    mass_phi_prime, decay_rate_phi_prime,
                                    mass_phi_double_prime, decay_rate_phi_double_prime,
                                    mass_rho, decay_rate_rho,
                                    mass_rho_prime, decay_rate_rho_prime,
                                    mass_rho_double_prime, decay_rate_rho_double_prime,
                                    mass_rho_triple_prime, decay_rate_rho_triple_prime
                                    )
    return partial_f


def make_partial_only_resonances_parameters(path_to_config, t_in_isoscalar, t_in_isovector,
                                            a_omega, a_omega_prime, a_omega_double_prime,
                                            a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime):
    t_0_isoscalar, t_0_isovector = _read_config(path_to_config)

    def partial_f(ts, mass_omega, decay_rate_omega, mass_omega_prime, decay_rate_omega_prime,
        mass_omega_double_prime, decay_rate_omega_double_prime, mass_phi, decay_rate_phi,
        mass_phi_prime, decay_rate_phi_prime, mass_phi_double_prime, decay_rate_phi_double_prime,
        mass_rho, decay_rate_rho, mass_rho_prime, decay_rate_rho_prime,
        mass_rho_double_prime, decay_rate_rho_double_prime, mass_rho_triple_prime, decay_rate_rho_triple_prime):
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector,
                                    t_in_isoscalar, t_in_isovector,
                                    a_omega, a_omega_prime, a_omega_double_prime,
                                    a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime,
                                    mass_omega, decay_rate_omega,
                                    mass_omega_prime, decay_rate_omega_prime,
                                    mass_omega_double_prime, decay_rate_omega_double_prime,
                                    mass_phi, decay_rate_phi,
                                    mass_phi_prime, decay_rate_phi_prime,
                                    mass_phi_double_prime, decay_rate_phi_double_prime,
                                    mass_rho, decay_rate_rho,
                                    mass_rho_prime, decay_rate_rho_prime,
                                    mass_rho_double_prime, decay_rate_rho_double_prime,
                                    mass_rho_triple_prime, decay_rate_rho_triple_prime
                                    )
    return partial_f


def get_bounds_low_energy(t_0_isoscalar, t_0_isovector):
    lower_mass_bound_isoscalar = math.sqrt(t_0_isoscalar)
    lower_mass_bound_isovector = math.sqrt(t_0_isovector)
    return (
        [t_0_isoscalar,  t_0_isovector, -np.inf, lower_mass_bound_isoscalar, 0.0,
         -np.inf, lower_mass_bound_isoscalar, 0.0, lower_mass_bound_isoscalar, 0.0,
         -np.inf, lower_mass_bound_isovector, 0.0, lower_mass_bound_isovector, 0.0],
        np.inf
    )


def make_partial_low_energy(
        path_to_config, a_omega_double_prime=0.0, mass_omega_double_prime=1.67, decay_rate_omega_double_prime=0.315,
        a_phi_prime=0.0, mass_phi_prime=1.680, decay_rate_phi_prime=0.150,
        mass_phi_double_prime=2.159, decay_rate_phi_double_prime=0.137,
        a_rho_double_prime=0.0, mass_rho_double_prime=1.720, decay_rate_rho_double_prime=0.25,
        mass_rho_triple_prime=2.15, decay_rate_rho_triple_prime=0.3):
    """
    Fit only the part of the model corresponding to the resonances:
        omega (0.78GeV), omega_prime (1.4GeV), phi (1.0GeV), rho (0.78GeV), rho_prime (1.5GeV).

    Set the coupling coefficients for the other particles to 0, unless otherwise specified.
    The coupling constants for rho_triple_prime and phi_double_prime are always fixed at zero.
    (Because of programming constraints, not for any physical reason.)

    """
    t_0_isoscalar, t_0_isovector = _read_config(path_to_config)

    def partial_f(ts, t_in_isoscalar, t_in_isovector,
                  a_omega, mass_omega, decay_rate_omega,
                  a_phi, mass_phi, decay_rate_phi,
                  mass_omega_prime, decay_rate_omega_prime,
                  a_rho, mass_rho, decay_rate_rho,
                  mass_rho_prime, decay_rate_rho_prime):
        a_omega_prime = 0.5 - a_omega - a_phi - a_omega_double_prime - a_phi_prime
        a_rho_prime = 0.5 - a_rho - a_rho_double_prime
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector,
                                    t_in_isoscalar, t_in_isovector,
                                    a_omega, a_omega_prime, a_omega_double_prime,
                                    a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime,
                                    mass_omega, decay_rate_omega,
                                    mass_omega_prime, decay_rate_omega_prime,
                                    mass_omega_double_prime, decay_rate_omega_double_prime,
                                    mass_phi, decay_rate_phi,
                                    mass_phi_prime, decay_rate_phi_prime,
                                    mass_phi_double_prime, decay_rate_phi_double_prime,
                                    mass_rho, decay_rate_rho,
                                    mass_rho_prime, decay_rate_rho_prime,
                                    mass_rho_double_prime, decay_rate_rho_double_prime,
                                    mass_rho_triple_prime, decay_rate_rho_triple_prime
                                    )
    return partial_f


def make_partial_high_energy(
        path_to_config, a_omega, mass_omega, decay_rate_omega,
        a_omega_prime, mass_omega_prime, decay_rate_omega_prime,
        a_phi, mass_phi, decay_rate_phi, a_rho, mass_rho, decay_rate_rho,
        a_rho_prime, mass_rho_prime, decay_rate_rho_prime):
    t_0_isoscalar, t_0_isovector = _read_config(path_to_config)

    def partial_f(ts, t_in_isoscalar, t_in_isovector,
                  a_omega_double_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
                  a_phi_prime, mass_phi_prime, decay_rate_phi_prime,
                  mass_phi_double_prime, decay_rate_phi_double_prime,
                  a_rho_double_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
                  mass_rho_triple_prime, decay_rate_rho_triple_prime):
        return function_form_factor(ts, t_0_isoscalar, t_0_isovector,
                                    t_in_isoscalar, t_in_isovector,
                                    a_omega, a_omega_prime, a_omega_double_prime,
                                    a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime,
                                    mass_omega, decay_rate_omega,
                                    mass_omega_prime, decay_rate_omega_prime,
                                    mass_omega_double_prime, decay_rate_omega_double_prime,
                                    mass_phi, decay_rate_phi,
                                    mass_phi_prime, decay_rate_phi_prime,
                                    mass_phi_double_prime, decay_rate_phi_double_prime,
                                    mass_rho, decay_rate_rho,
                                    mass_rho_prime, decay_rate_rho_prime,
                                    mass_rho_double_prime, decay_rate_rho_double_prime,
                                    mass_rho_triple_prime, decay_rate_rho_triple_prime
                                    )
    return partial_f
