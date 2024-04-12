from configparser import ConfigParser
from typing import Callable, Tuple
from common.utils import make_partial_form_factor_for_parameters
from model_parameters import KaonParametersSimplified, Parameter, PionParameters
from kaon_production.data import KaonDatapoint
from ua_model.PionUAModel import PionUAModel

import math
import numpy
from scipy.integrate import quad


def _wrap_partial_form_factor_function(partial_f: Callable, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=False)
        res = partial_f([datapoint])[0].real
        return res
    return wrapped


def _make_partial_for_pion_parameters(pion_parameters: PionParameters) -> Callable:
    pion_parameters = pion_parameters.copy()
    ff_model = PionUAModel(**{p.name: p.value for p in pion_parameters})

    def wrapped(s):
        res = ff_model(s).real
        return res
    return wrapped


def calculate_em_mass2_contribution(form_factor_function, alpha, particle_mass) -> Tuple[float, float]:
    """
    Equation (39) of https://doi.org/10.1140/epjc/s10052-022-10348-3
    (Stamen et al., Kaon electromagnetic form factors in dispersion theory)

    """
    coefficient = alpha / (8 * math.pi)
    mass_squared = particle_mass**2

    def integrand(s):
        w = math.sqrt(1 + 4 * mass_squared / s)
        return (form_factor_function(-s)**2) * (4 * w + s * (w - 1) / mass_squared)

    integral_val = quad(integrand, 0, numpy.inf)
    return coefficient * integral_val[0], coefficient * integral_val[1]


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')

    for i in [2, 29, 40, 125, 151]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/run9simplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=True
        )
        em_mass2_charged_kaon = calculate_em_mass2_contribution(f, alpha, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} EM mass^2 charged kaon: {em_mass2_charged_kaon}')

        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=False
        )
        em_mass2_neutral_kaon = calculate_em_mass2_contribution(f, alpha, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} EM mass^2 neutral kaon: {em_mass2_neutral_kaon}')

    for i in [37, 113, 160, 165]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runTestDressedsimplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=True
        )
        em_mass2_charged_kaon = calculate_em_mass2_contribution(f, alpha, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} EM mass^2 charged kaon: {em_mass2_charged_kaon}')

        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=False
        )
        em_mass2_neutral_kaon = calculate_em_mass2_contribution(f, alpha, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} EM mass^2 neutral kaon: {em_mass2_neutral_kaon}')

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
    pion_em_m2 = calculate_em_mass2_contribution(f_pion, alpha, charged_kaon_mass)
    print(f'EM mass^2 charged pion: {pion_em_m2}')

