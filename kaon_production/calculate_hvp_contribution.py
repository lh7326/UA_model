from configparser import ConfigParser
from typing import Callable, Tuple
from common.utils import make_partial_cross_section_for_parameters
from model_parameters import KaonParametersSimplified
from kaon_production.data import KaonDatapoint
from kaon_production.eta_correction import add_fsr_effects

from ua_model.PionUAModel import PionUAModel
from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection

import math
from scipy.integrate import quad


def _kernel(s, muon_mass):
    beta = math.sqrt(1.0 - 4 * (muon_mass**2) / s)
    x = (1.0 - beta) / (1.0 + beta)
    x2 = x**2
    return 0.5 * x2 * (2.0 - x2) + (1.0 + x2) * ((1.0 + x)**2) * (
            math.log(1.0 + x) - x + 0.5 * x2) / x2 + ((1.0 + x) / (1.0 - x)) * x2 * math.log(x)


def _kernel_integral_representation(s, muon_mass):
    m2 = muon_mass**2

    def integrand(x):
        x2 = x**2
        return x2 * (1.0 - x) / (x2 + (s / m2) * (1.0 - x))

    return quad(integrand, 0, 1)


def _wrap_partial_cross_section_function(
        partial_f: Callable, charged_kaon_mass: float, alpha: float, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=True)
        res = partial_f([datapoint])[0].real
        if charged:
            return add_fsr_effects(res, s, charged_kaon_mass, alpha)
        return res
    return wrapped


def calculate_hvp_contribution(cross_section_function, alpha, hc_squared, muon_mass, kaon_mass) -> Tuple[float, float]:
    coefficient = alpha**2 / (3.0 * (math.pi**2))

    def integrand(s):
        r = cross_section_function(s) / (4.0 * hc_squared * math.pi * alpha**2 / (3.0 * s))
        return _kernel(s, muon_mass) * r / s

    integral_val = quad(integrand, 4 * (kaon_mass**2), 1.05**2)
    return coefficient * integral_val[0], coefficient * integral_val[1]


def make_calculate_hvp_contribution_kaon(
        alpha, hc_squared, muon_mass, charged_kaon_mass, neutral_kaon_mass, charged
):
    def f(parameters):
        parameters = parameters.copy()
        parameters.fix_all_parameters()
        partial = make_partial_cross_section_for_parameters(
            alpha, hc_squared, parameters,
            charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass
        )
        cs_f = _wrap_partial_cross_section_function(partial, charged_kaon_mass, alpha, charged)
        kaon_mass = charged_kaon_mass if charged else neutral_kaon_mass
        return calculate_hvp_contribution(cs_f, alpha, hc_squared, muon_mass, kaon_mass)[0]
    return f


def make_calculate_hvp_contribution_pion(alpha, hc_squared, muon_mass, charged_pion_mass,
                                         upper_limit=1.05):
    def _make_ff_function(pion_parameters):
        pion_parameters = pion_parameters.copy()
        return PionUAModel(**{p.name: p.value for p in pion_parameters})

    def _make_cs_function(ff_fun):
        config = ConfigParser()
        config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}
        return ScalarMesonProductionTotalCrossSection(charged_pion_mass, ff_fun, config)

    def _calculate_hvp_contribution(cross_section_function):
        coefficient = alpha ** 2 / (3.0 * (math.pi ** 2))

        def integrand(s):
            r = cross_section_function(s) / (4.0 * hc_squared * math.pi * alpha ** 2 / (3.0 * s))
            return _kernel(s, muon_mass) * r / s

        integral_val = quad(integrand, 4 * (charged_pion_mass ** 2), upper_limit ** 2)
        return coefficient * integral_val[0]

    def f(parameters):
        parameters = parameters.copy()
        ff_fun = _make_ff_function(parameters)
        cs_f = _make_cs_function(ff_fun)
        return _calculate_hvp_contribution(cs_f)

    return f


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    muon_mass = config.getfloat('constants', 'muon_mass')

    for i in [2, 29, 40, 125, 151]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/run9simplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged_kaon_mass, alpha,
            charged=True,
        )
        charged_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} HVP contribution charged kaon: {charged_kaon_hvp_contribution}')

        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged_kaon_mass, alpha,
            charged=False,
        )
        neutral_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} HVP contribution neutral kaon: {neutral_kaon_hvp_contribution}')

    for i in [37, 113, 160, 165]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runTestDressedsimplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged_kaon_mass, alpha,
            charged=True,
        )
        charged_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} HVP contribution charged kaon: {charged_kaon_hvp_contribution}')

        f = _wrap_partial_cross_section_function(
            make_partial_cross_section_for_parameters(
                alpha, hc_squared, kaon_parameters,
                charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass),
            charged_kaon_mass, alpha,
            charged=False,
        )
        neutral_kaon_hvp_contribution = calculate_hvp_contribution(f, alpha, hc_squared, muon_mass, charged_kaon_mass)
        print(f'{kaon_parameters_filepath} HVP contribution neutral kaon: {neutral_kaon_hvp_contribution}')
