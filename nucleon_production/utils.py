from configparser import ConfigParser
import random
from typing import Optional, Callable, Tuple, Union

import numpy as np

from nucleon_production.function import function_nucleon_cross_section, function_nucleon_form_factor
from model_parameters import ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters


def make_partial_cross_section_for_parameters(
        parameters: NucleonParameters
) -> Callable:

    # create a local copy of the parameters
    parameters = NucleonParameters.from_list(parameters.to_list())

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_nucleon_cross_section(ts)

    return partial_f


def make_partial_form_factor_for_parameters(
        parameters: Union[
            ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters
        ]
) -> Callable:

    # TODO: implement a copy method on the level of ModelParameters
    # create a local copy of the parameters
    if isinstance(parameters, ETGMRModelParameters):
        parameters = ETGMRModelParameters.from_list(parameters.to_list())
    elif isinstance(parameters, TwoPolesModelParameters):
        parameters = TwoPolesModelParameters.from_list(parameters.to_list())
    elif isinstance(parameters, NucleonParameters):
        parameters = NucleonParameters.from_list(parameters.to_list())
    else:
        TypeError('Unexpected parameters type: ' + type(parameters).__name__)

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_nucleon_form_factor(ts, parameters)

    return partial_f


def _read_config(path_to_config: str) -> Tuple[float, float]:
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector
