from configparser import ConfigParser
import random
from typing import Optional, Callable, Tuple

import numpy as np

from kaon_production.function import function_form_factor, function_cross_section
from kaon_production.ModelParameters import ModelParameters


def make_partial_form_factor_for_parameters(parameters: ModelParameters) -> Callable:
    def _build_parameters_scheme():
        argument_index = 0
        scheme = []
        for parameter in parameters:
            if parameter.is_fixed:
                scheme.append(lambda _, p=parameter: p.value)
            else:
                scheme.append(lambda args, i=argument_index: args[i])
                argument_index += 1
        return scheme, argument_index

    scheme, args_length = _build_parameters_scheme()

    def partial_f(ts, *args):
        assert len(args) == args_length
        evaluated_scheme = [val(args) for val in scheme]
        return function_form_factor(
            ts, parameters.t_0_isoscalar, parameters.t_0_isovector, *evaluated_scheme
        )

    return partial_f


def make_partial_cross_section_for_parameters(k_meson_mass: float, alpha: float, hc_squared: float,
                                              parameters: ModelParameters) -> Callable:
    def _build_parameters_scheme():
        argument_index = 0
        scheme = []
        for parameter in parameters:
            if parameter.is_fixed:
                scheme.append(lambda _, p=parameter: p.value)
            else:
                scheme.append(lambda args, i=argument_index: args[i])
                argument_index += 1
        return scheme, argument_index

    scheme, args_length = _build_parameters_scheme()

    def partial_f(ts, *args):
        assert len(args) == args_length
        evaluated_scheme = [val(args) for val in scheme]
        return function_cross_section(
            ts, k_meson_mass, alpha, hc_squared,
            parameters.t_0_isoscalar, parameters.t_0_isovector, *evaluated_scheme
        )

    return partial_f


def _read_config(path_to_config: str) -> Tuple[float, float]:
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector


def perturb_model_parameters(
        parameters: ModelParameters,
        perturbation_size: float = 0.2,
        perturbation_size_resonances: Optional[float] = None,
        respect_fixed: bool = False,
        use_handpicked_bounds: bool = True,
        ) -> ModelParameters:
    if perturbation_size_resonances is None:
        perturbation_size_resonances = perturbation_size

    def _get_perturbation_size(parameter_name):
        if 'mass' in parameter_name or 'decay_rate' in parameter_name:
            return perturbation_size_resonances
        else:
            return perturbation_size

    def _get_perturbed_value(lower_bound, old_value, upper_bound, rnd, ps):
        if rnd < 0:
            if lower_bound == -np.inf:
                return old_value + ps * old_value * rnd
            else:
                return old_value + ps * (old_value - lower_bound) * rnd
        elif rnd > 0:
            if upper_bound == np.inf:
                return old_value + ps * old_value * rnd
            else:
                return old_value + ps * (upper_bound - old_value) * rnd
        else:
            return old_value

    if use_handpicked_bounds:
        bounds = parameters.get_model_parameters_bounds_handpicked()
    else:
        bounds = parameters.get_model_parameters_bounds_maximal()

    for p in parameters:
        if respect_fixed and p.is_fixed:
            continue
        random_number = 2 * (random.random() - 0.5)  # the interval [-1, +1)
        lower_bound = bounds[p.name]['lower']
        upper_bound = bounds[p.name]['upper']
        perturbed_value = _get_perturbed_value(
            lower_bound, p.value, upper_bound,
            random_number, _get_perturbation_size(p.name),
        )
        parameters.set_value(p.name, perturbed_value)
    return parameters
