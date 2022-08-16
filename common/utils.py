import random
from typing import Union, Optional

import numpy as np

from model_parameters import KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega, \
    KaonParametersFixedSelected, ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters


def perturb_model_parameters(
        parameters: Union[
            KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega,
            KaonParametersFixedSelected, ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters],
        perturbation_size: float = 0.2,
        perturbation_size_resonances: Optional[float] = None,
        use_handpicked_bounds: bool = True,
        ) -> KaonParameters:
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
                return -1 * abs(old_value) * (1 + ps * abs(rnd))
            else:
                return old_value + ps * (old_value - lower_bound) * rnd
        elif rnd > 0:
            if upper_bound == np.inf:
                return abs(old_value) * (1 + ps * abs(rnd))
            else:
                return old_value + ps * (upper_bound - old_value) * rnd
        else:
            return old_value

    if use_handpicked_bounds:
        bounds = parameters.get_model_parameters_bounds_handpicked()
    else:
        bounds = parameters.get_model_parameters_bounds_maximal()

    for p in parameters:
        if p.is_fixed:
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
