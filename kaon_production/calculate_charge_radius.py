from configparser import ConfigParser
from typing import Callable
from common.utils import make_partial_form_factor_for_parameters
from model_parameters import KaonParametersFixedSelected, Parameter, PionParameters
from kaon_production.data import KaonDatapoint
from ua_model.PionUAModel import PionUAModel

from scipy.misc import derivative


# TODO: Use a different implementation. (scipy.misc.derivative is deprecated)

def _wrap_partial_form_factor_function(partial_f: Callable, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=False)
        res = partial_f([datapoint])[0].real
        label = 'charged' if charged else 'neutral'
        print(f'ff_kaon_{label}({s})={res}')
        return res
    return wrapped


def _make_partial_for_pion_parameters(pion_parameters: PionParameters) -> Callable:
    pion_parameters = pion_parameters.copy()
    ff_model = PionUAModel(**{p.name: p.value for p in pion_parameters})

    def wrapped(s):
        res = ff_model(s).real
        print(f'ff_pion({s})={res}')
        return res
    return wrapped


def calculate_charge_radius(form_factor_function, hc_squared) -> float:
    return 6.0 * hc_squared * derivative(form_factor_function, x0=0.0, dx=1e-6, n=1, order=9)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    hc_squared = config.getfloat('constants', 'hc_squared')
    # this is in GeV^2 * nanobarn
    # to obtain the result in fm^2 we need to multiply by 10^-7,
    # because 1nb = 10^-37m^2 and 1fm^2 = 10^-30m^2
    hc_squared = hc_squared * 1e-7

    # kaon_parameters_filepath = '/home/lukas/reports/kaons/run2/run2_6/final_fit_parameters.pickle'
    # kaon_parameters_filepath = '/home/lukas/reports/kaons/run3/run3_0/final_fit_parameters.pickle'
    # kaon_parameters_filepath = '/home/lukas/reports/kaons/run4/run4_0/final_fit_parameters.pickle'
    # kaon_parameters_filepath = '/home/lukas/reports/kaons/run4/run4_29/final_fit_parameters.pickle'
    # kaon_parameters_filepath = '/home/lukas/reports/kaons/run5_56/final_fit_parameters.pickle'
    # kaon_parameters = KaonParametersFixedSelected.load_from_serialized_parameters(kaon_parameters_filepath)
    # kaon_parameters.fix_all_parameters()
    # f = _wrap_partial_form_factor_function(
    #     make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
    #     charged=True
    # )
    # charge_radius_charged = calculate_charge_radius(f, hc_squared)
    # print(f'Charge radius charged kaon: {charge_radius_charged}')
    #
    # f = _wrap_partial_form_factor_function(
    #     make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
    #     charged=False
    # )
    # charge_radius_neutral = calculate_charge_radius(f, hc_squared)
    # print(f'Charge radius neutral kaon: {charge_radius_neutral}')

    pion_parameters = PionParameters.from_list([
        Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
        Parameter(name='t_in_isovector', value=1.2733, is_fixed=False),
        Parameter(name='mass_rho', value=0.7621, is_fixed=False),
        Parameter(name='decay_rate_rho', value=0.14423672, is_fixed=False),
        Parameter(name='a_rho_prime', value=-0.07060638, is_fixed=False),
        Parameter(name='mass_rho_prime', value=1.3500, is_fixed=False),
        Parameter(name='decay_rate_rho_prime', value=0.3319913, is_fixed=False),
        Parameter(name='a_rho_double_prime', value=0.05785514, is_fixed=False),
        Parameter(name='mass_rho_double_prime', value=1.76928672, is_fixed=False),
        Parameter(name='decay_rate_rho_double_prime', value=0.25311443, is_fixed=False),
        Parameter(name='a_rho_triple_prime', value=0.00208887, is_fixed=False),
        Parameter(name='mass_rho_triple_prime', value=2.24674832, is_fixed=False),
        Parameter(name='decay_rate_rho_triple_prime', value=0.0700, is_fixed=False),
        Parameter(name='w_pole', value=0.38329263, is_fixed=False),
        Parameter(name='w_zero', value=0.2844582, is_fixed=False),
    ])
    f_pion = _make_partial_for_pion_parameters(pion_parameters)
    pion_charge_radius = calculate_charge_radius(f_pion, hc_squared)
    print(f'Charge radius charged pion: {pion_charge_radius}')

    test_pion_parameters = pion_parameters.copy()
    ff_model_pions = PionUAModel(**{p.name: p.value for p in pion_parameters})
    def f(s):
        res = ff_model_pions(s)
        print(f'ff_pion({s})={res}')
        return res

    test_values = [0.0, 0.05, 0.1, 0.5, 1.0, 1.2, 1.5, 2.0]
    for s in test_values:
        f(s)
