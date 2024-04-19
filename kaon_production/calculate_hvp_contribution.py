from configparser import ConfigParser
from typing import Callable, Tuple
from common.utils import make_partial_cross_section_for_parameters
from model_parameters import KaonParametersSimplified
from kaon_production.data import KaonDatapoint

import math
import numpy
from scipy.integrate import quad


def _kernel(s, muon_mass):
    beta = math.sqrt(1.0 - 4 * (muon_mass**2) / s)
    x = (1.0 - beta) / (1.0 + beta)
    x2 = x**2
    return 0.5 * x2 * (2.0 - x2) + (1.0 + x2) * ((1.0 + x)**2) * (
            math.log(1.0 + x) - x + 0.5 * x2) / x2 + ((1.0 + x) / (1.0 - x)) * x2 * math.log(x)


def _wrap_partial_cross_section_function(partial_f: Callable, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=True)
        res = partial_f([datapoint])[0].real
        return res
    return wrapped


def calculate_hvp_contribution(cross_section_function, alpha, hc_squared, muon_mass, pion_mass) -> Tuple[float, float]:
    coefficient = alpha**2 / (3.0 * (math.pi**2))

    def integrand(s):
        r = cross_section_function(s) / (4.0 * hc_squared * math.pi * alpha**2 / (3.0 * s))
        return _kernel(s, muon_mass) * r / s**2

    integral_val = quad(integrand, 4*pion_mass**2, 1.05**2)
    return coefficient * integral_val[0], coefficient * integral_val[1]


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    neutral_pion_mass = config.getfloat('constants', 'neutral_pion_mass')
    muon_mass = config.getfloat('constants', 'muon_mass')

    for i in [2, 29, 40, 125, 151]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/run9simplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged=True,
        )
        charged_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, neutral_pion_mass)
        print(f'{kaon_parameters_filepath} HVP contribution charged kaon: {charged_kaon_hvp_contribution}')

        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged=False,
        )
        neutral_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, neutral_pion_mass)
        print(f'{kaon_parameters_filepath} HVP contribution neutral kaon: {neutral_kaon_hvp_contribution}')

    for i in [37, 113, 160, 165]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runTestDressedsimplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged=True,
        )
        charged_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, neutral_pion_mass)
        print(f'{kaon_parameters_filepath} HVP contribution charged kaon: {charged_kaon_hvp_contribution}')

        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged=False,
        )
        neutral_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, neutral_pion_mass)
        print(f'{kaon_parameters_filepath} HVP contribution neutral kaon: {neutral_kaon_hvp_contribution}')
