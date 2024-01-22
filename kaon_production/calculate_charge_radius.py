from configparser import ConfigParser
from typing import Callable
from common.utils import make_partial_form_factor_for_parameters
from model_parameters import KaonParametersFixedSelected, Parameter
from kaon_production.data import KaonDatapoint

from scipy.misc import derivative


# TODO: Use a different implementation. (scipy.misc.derivative is deprecated)

def _wrap_partial_form_factor_function(partial_f: Callable, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=False)
        return partial_f([datapoint])[0]
    return wrapped


def calculate_charge_radius(form_factor_function, hc_squared) -> float:
    return 6.0 * hc_squared * derivative(form_factor_function, x0=0.0, dx=1e-7, n=1, order=9)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    hc_squared = config.getfloat('constants', 'hc_squared')
    # this is in GeV^2 * nanobarn
    # to obtain the result in fm^2 we need to multiply by 10^7,
    # because 1nb = 10^-37m^2 and 1fm^2 = 10^-30m^2
    hc_squared = hc_squared * 1e7


    parameters = KaonParametersFixedSelected.from_list([
        Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
        Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
        Parameter(name='t_in_isoscalar', value=2.6248591992808086, is_fixed=False),
        Parameter(name='t_in_isovector', value=2.435372806237077, is_fixed=False),
        Parameter(name='a_omega', value=0.15792834419710358, is_fixed=False),
        Parameter(name='mass_omega', value=0.78266, is_fixed=True),
        Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
        Parameter(name='a_omega_prime', value=-0.04626459159918863, is_fixed=False),
        Parameter(name='mass_omega_prime', value=1.42, is_fixed=True),
        Parameter(name='decay_rate_omega_prime', value=0.2569864367626886, is_fixed=False),
        Parameter(name='a_omega_double_prime', value=-0.8914000357539935, is_fixed=False),
        Parameter(name='mass_omega_double_prime', value=1.67, is_fixed=True),
        Parameter(name='decay_rate_omega_double_prime', value=0.5060759599350851, is_fixed=False),
        Parameter(name='a_phi', value=-0.30479413283897105, is_fixed=False),
        Parameter(name='mass_phi', value=1.0190519994842593, is_fixed=False),
        Parameter(name='decay_rate_phi', value=0.004072613361751101, is_fixed=False),
        Parameter(name='a_phi_prime', value=0.20525835231060238, is_fixed=False),
        Parameter(name='mass_phi_prime', value=1.6671564479506489, is_fixed=False),
        Parameter(name='decay_rate_phi_prime', value=0.22993288602474032, is_fixed=False),
        Parameter(name='mass_phi_double_prime', value=1.6201417291657372, is_fixed=False),
        Parameter(name='decay_rate_phi_double_prime', value=0.9223291776336039, is_fixed=False),
        Parameter(name='a_rho', value=0.27929055741172604, is_fixed=False),
        Parameter(name='mass_rho', value=0.75823, is_fixed=True),
        Parameter(name='decay_rate_rho', value=0.14456, is_fixed=True),
        Parameter(name='a_rho_prime', value=-0.039660551355615255, is_fixed=False),
        Parameter(name='mass_rho_prime', value=1.0191360026442808, is_fixed=False),
        Parameter(name='decay_rate_rho_prime', value=0.005445846827360469, is_fixed=False),
        Parameter(name='a_rho_double_prime', value=0.02545201464878091, is_fixed=False),
        Parameter(name='mass_rho_double_prime', value=1.1101806912123264, is_fixed=False),
        Parameter(name='decay_rate_rho_double_prime', value=0.11478188288588474, is_fixed=False),
        Parameter(name='mass_rho_triple_prime', value=1.5605681037417838, is_fixed=False),
        Parameter(name='decay_rate_rho_triple_prime', value=1.0437469441524068, is_fixed=False)
    ])
    parameters.fix_all_parameters()
    f = _wrap_partial_form_factor_function(
        make_partial_form_factor_for_parameters(parameters),
        charged=True
    )
    charge_radius_charged = calculate_charge_radius(f, hc_squared)
    print(f'Charge radius charged kaon: {charge_radius_charged}')

    f = _wrap_partial_form_factor_function(
        make_partial_form_factor_for_parameters(parameters),
        charged=False
    )
    charge_radius_neutral = calculate_charge_radius(f, hc_squared)
    print(f'Charge radius neutral kaon: {charge_radius_neutral}')
