import random
from configparser import ConfigParser
from typing import Callable, List, Tuple, Union, Optional, TypeVar

import numpy as np

from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from cross_section.NucleonPairToElectronPositronTotalCrossSection import NucleonPairToElectronPositronTotalCrossSection
from ua_model.KaonUAModel import KaonUAModel
from ua_model.KaonUAModelSimplified import KaonUAModelSimplified
from ua_model.KaonUAModelB import KaonUAModelB
from ua_model.NucleonUAModel import NucleonUAModel
from other_models import ETGMRModel, TwoPolesModel
from model_parameters import (ModelParameters, KaonParameters, KaonParametersB, KaonParametersSimplified,
                              KaonParametersFixedRhoOmega, KaonParametersFixedSelected, ETGMRModelParameters,
                              TwoPolesModelParameters, NucleonParameters)
from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint


T = TypeVar(
    'T', KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega,
    KaonParametersFixedSelected, ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters,
)


def perturb_model_parameters(
        parameters: T,
        perturbation_size: float = 0.2,
        perturbation_size_resonances: Optional[float] = None,
        use_handpicked_bounds: bool = True,
        ) -> T:
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


def _get_ff_model(
        parameters: ModelParameters,
) -> Union[KaonUAModel, KaonUAModelB, KaonUAModelSimplified, ETGMRModel, TwoPolesModel, NucleonUAModel]:
    if isinstance(parameters, KaonParameters):
        return KaonUAModel(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersB):
        return KaonUAModelB(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersSimplified):
        return KaonUAModelSimplified(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersFixedRhoOmega):
        return KaonUAModel(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersFixedSelected):
        return KaonUAModel(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, NucleonParameters):
        return NucleonUAModel(proton=True, electric=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, ETGMRModelParameters):
        return ETGMRModel(a=parameters['a'].value, m_a=parameters['m_a'].value, m_d=parameters['m_d'].value)
    elif isinstance(parameters, TwoPolesModelParameters):
        return TwoPolesModel(a=parameters['a'].value, m_1=parameters['m_1'].value, m_2=parameters['m_2'].value)
    else:
        raise TypeError('Unexpected parameters type: ' + type(parameters).__name__)


def _read_datapoint_kaon(datapoint: Union[KaonDatapoint, Tuple[float, float, float]]) -> Tuple[float, bool, bool]:
    if isinstance(datapoint, KaonDatapoint):
        return datapoint.t, datapoint.is_charged, datapoint.is_for_cross_section
    else:
        return float(datapoint[0]), bool(datapoint[1]), bool(datapoint[2])


def _read_datapoint_nucleon(
        datapoint: Union[NucleonDatapoint, Tuple[float, float, float]]
) -> Tuple[float, bool, bool]:
    if isinstance(datapoint, NucleonDatapoint):
        return datapoint.t, datapoint.proton, datapoint.electric
    else:
        return float(datapoint[0]), bool(datapoint[1]), bool(datapoint[2])


def _is_kaon_type_model(ff_model: Callable) -> bool:
    if isinstance(ff_model, (KaonUAModel, KaonUAModelB, KaonUAModelSimplified)):
        return True
    elif isinstance(ff_model, (NucleonUAModel, ETGMRModel, TwoPolesModel)):
        return False
    else:
        raise f'Unknown model: {type(ff_model)}!'


def function_form_factor(
        ts: Union[
            List[Union[KaonDatapoint, Tuple[float, float, float]]],
            List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        ],
        parameters: ModelParameters,
        return_absolute_value: bool,
        ) -> List[float]:

    ff_model = _get_ff_model(parameters)

    is_kaon_type = _is_kaon_type_model(ff_model)
    results = []
    if is_kaon_type:
        for datapoint in ts:
            t, is_charged, is_for_cross_section = _read_datapoint_kaon(datapoint)
            if is_for_cross_section:
                raise ValueError('Datapoint {datapoint} is for cross section, not form factor!')
            ff_model.charged_variant = is_charged
            res = ff_model(t)
            if return_absolute_value:
                results.append(abs(res))
            else:
                results.append(res)
    else:  # a nucleon form factor model
        for datapoint in ts:
            t, is_proton, is_electric = _read_datapoint_nucleon(datapoint)  # type: ignore
            ff_model.proton = is_proton
            ff_model.electric = is_electric
            res = ff_model(t)
            if return_absolute_value:
                results.append(abs(res))
            else:
                results.append(res)

    return results


def function_cross_section(
        ts: Union[
            List[Union[KaonDatapoint, Tuple[float, float, float]]],
            List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        ],
        product_particle_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: ModelParameters,
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    config = ConfigParser()
    config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}
    if _is_kaon_type_model(ff_model):
        cross_section_model = ScalarMesonProductionTotalCrossSection(
            product_particle_mass, ff_model, config)
    elif isinstance(ff_model, (ETGMRModel, TwoPolesModel)):
        # In these cases the model can describe (with suitable parameters)
        # both form factors and cross-sections
        cross_section_model = ff_model
    else:
        cross_section_model = NucleonPairToElectronPositronTotalCrossSection(
            product_particle_mass, ff_model, config
        )

    results = []
    # TODO: refactor
    if _is_kaon_type_model(ff_model):
        for datapoint in ts:
            t, is_charged, is_for_cross_section = _read_datapoint_kaon(datapoint)
            if not is_for_cross_section:
                raise ValueError('Datapoint {datapoint} is for form factor, not cross section!')
            cross_section_model.form_factor.charged_variant = is_charged
            results.append(abs(cross_section_model(t)))
    else:  # a nucleon form factor model
        for datapoint in ts:
            t, is_proton, _ = _read_datapoint_nucleon(datapoint)  # type: ignore
            cross_section_model.form_factor.proton = is_proton
            results.append(abs(cross_section_model(t)))

    return results


# TODO: refactor!
def function_form_factor_or_cross_section(
        xs: Union[
            List[Union[KaonDatapoint, Tuple[float, float, float]]],
            List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        ],
        product_particle_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: ModelParameters,
        ) -> List[float]:

    ff_model = _get_ff_model(parameters)
    config = ConfigParser()
    config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}

    is_kaon_type = _is_kaon_type_model(ff_model)
    if is_kaon_type:
        cs_model = ScalarMesonProductionTotalCrossSection(
            product_particle_mass, ff_model, config)
    elif isinstance(ff_model, (ETGMRModel, TwoPolesModel)):
        # In these cases the model can describe (with suitable parameters)
        # both form factors and cross-sections
        cs_model = ff_model
    else:
        cs_model = NucleonPairToElectronPositronTotalCrossSection(
            product_particle_mass, ff_model, config
        )

    results = []
    if is_kaon_type:
        for datapoint in xs:
            t, is_charged, is_for_cross_section = _read_datapoint_kaon(datapoint)
            if is_for_cross_section:
                cs_model.form_factor.charged_variant = is_charged
                results.append(abs(cs_model(t)))
            else:
                ff_model.charged_variant = is_charged
                results.append(abs(ff_model(t)))
    else:  # a nucleon form factor model
        for datapoint in xs:
            is_for_cross_section = True  # TODO: allow both cs and ff cases
            t, is_proton, is_electric = _read_datapoint_nucleon(datapoint)  # type: ignore
            if is_for_cross_section:
                cs_model.form_factor.proton = is_proton
                results.append(abs(cs_model(t)))
            else:
                ff_model.proton = is_proton
                ff_model.electric = is_electric
                results.append(abs(ff_model(t)))

    return results


def make_partial_form_factor_for_parameters(
        parameters: ModelParameters,
        return_absolute_value: bool = True,
) -> Callable:

    parameters = parameters.copy()

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_form_factor(ts, parameters, return_absolute_value)

    return partial_f


def make_partial_cross_section_for_parameters(
        product_particle_mass: float, alpha: float, hc_squared: float,
        parameters: ModelParameters,
) -> Callable:
    parameters = parameters.copy()

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_cross_section(ts, product_particle_mass, alpha, hc_squared, parameters)

    return partial_f


def make_partial_ff_or_cs_for_parameters(
        product_particle_mass: float, alpha: float, hc_squared: float,
        parameters: ModelParameters,
) -> Callable:
    parameters = parameters.copy()

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_form_factor_or_cross_section(ts, product_particle_mass, alpha, hc_squared, parameters)

    return partial_f
