from configparser import ConfigParser
from typing import Callable, Tuple, Union

from kaon_production.function import function_kaon_cross_section, function_kaon_form_factor
from model_parameters import (KaonParameters, KaonParametersSimplified,
                              KaonParametersFixedRhoOmega, KaonParametersFixedSelected, KaonParametersB,
                              ETGMRModelParameters, TwoPolesModelParameters)


def make_partial_cross_section_for_parameters(
        k_meson_mass: float, alpha: float, hc_squared: float,
        parameters: Union[
            KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega,
            KaonParametersFixedSelected]
) -> Callable:

    # create a local copy of the parameters
    if isinstance(parameters, KaonParameters):
        parameters = KaonParameters.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersB):
        parameters = KaonParametersB.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersSimplified):
        parameters = KaonParametersSimplified.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersFixedRhoOmega):
        parameters = KaonParametersFixedRhoOmega.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersFixedSelected):
        parameters = KaonParametersFixedSelected.from_list(parameters.to_list())
    else:
        TypeError('Unexpected parameters type: ' + type(parameters).__name__)

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_kaon_cross_section(ts, k_meson_mass, alpha, hc_squared, parameters)

    return partial_f


def make_partial_form_factor_for_parameters(
        parameters: Union[
            KaonParameters, KaonParametersB, KaonParametersFixedRhoOmega, KaonParametersFixedSelected,
            ETGMRModelParameters, TwoPolesModelParameters
        ]
) -> Callable:

    # TODO: implement a copy method on the level of ModelParameters
    # create a local copy of the parameters
    if isinstance(parameters, KaonParameters):
        parameters = KaonParameters.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersB):
        parameters = KaonParametersB.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersFixedRhoOmega):
        parameters = KaonParametersFixedRhoOmega.from_list(parameters.to_list())
    elif isinstance(parameters, KaonParametersFixedSelected):
        parameters = KaonParametersFixedSelected.from_list(parameters.to_list())
    elif isinstance(parameters, ETGMRModelParameters):
        parameters = ETGMRModelParameters.from_list(parameters.to_list())
    elif isinstance(parameters, TwoPolesModelParameters):
        parameters = TwoPolesModelParameters.from_list(parameters.to_list())
    else:
        TypeError('Unexpected parameters type: ' + type(parameters).__name__)

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_kaon_form_factor(ts, parameters)

    return partial_f


def _read_config(path_to_config: str) -> Tuple[float, float]:
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector
