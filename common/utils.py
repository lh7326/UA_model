import random
from configparser import ConfigParser
from typing import Callable, List, Tuple, Union, Optional, TypeVar

import numpy as np

from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from cross_section.NucleonPairToElectronPositronTotalCrossSection import NucleonPairToElectronPositronTotalCrossSection
from ua_model.KaonUAModel import KaonUAModel
from ua_model.KaonUAModelSimplified import KaonUAModelSimplified
from ua_model.KaonUAModelB import KaonUAModelB
from ua_model.KaonUAModelPhiRatio import KaonUAModelPhiRatio
from ua_model.NucleonUAModel import NucleonUAModel
from other_models import ETGMRModel, TwoPolesModel
from model_parameters import (ModelParameters, KaonParameters, KaonParametersB, KaonParametersSimplified,
                              KaonParametersFixedRhoOmega, KaonParametersFixedSelected, KaonParametersPhiRatio,
                              ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters)
from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint


T = TypeVar(
    'T', KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega,
    KaonParametersFixedSelected, KaonParametersPhiRatio, ETGMRModelParameters, TwoPolesModelParameters,
    NucleonParameters,
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
) -> Union[KaonUAModel, KaonUAModelB, KaonUAModelSimplified, KaonUAModelPhiRatio,
           ETGMRModel, TwoPolesModel, NucleonUAModel]:
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
    elif isinstance(parameters, KaonParametersPhiRatio):
        return KaonUAModelPhiRatio(charged_variant=True, **{p.name: p.value for p in parameters})
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
    if isinstance(ff_model, (KaonUAModel, KaonUAModelB, KaonUAModelSimplified, KaonUAModelPhiRatio)):
        return True
    elif isinstance(ff_model, (NucleonUAModel, ETGMRModel, TwoPolesModel)):
        return False
    else:
        raise f'Unknown model: {type(ff_model)}!'


def _are_kaon_parameters(parameters: ModelParameters) -> bool:
    return isinstance(
        parameters,
        (KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersPhiRatio,
         KaonParametersFixedRhoOmega, KaonParametersFixedSelected)
    )


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


def function_kaon_cross_section(
        ts:  List[Union[KaonDatapoint, Tuple[float, float, float]]],
        charged_kaon_mass: float,
        neutral_kaon_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: ModelParameters,
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    config = ConfigParser()
    config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}
    if _is_kaon_type_model(ff_model):
        cross_section_model_charged = ScalarMesonProductionTotalCrossSection(
            charged_kaon_mass, ff_model, config)
        cross_section_model_neutral = ScalarMesonProductionTotalCrossSection(
            neutral_kaon_mass, ff_model, config)
    else:
        raise ValueError(f'Unexpected parameter type: {type(parameters)}')

    results = []
    for datapoint in ts:
        t, is_charged, is_for_cross_section = _read_datapoint_kaon(datapoint)
        if not is_for_cross_section:
            raise ValueError('Datapoint {datapoint} is for form factor, not cross section!')
        if is_charged:
            cross_section_model_charged.form_factor.charged_variant = is_charged
            results.append(abs(cross_section_model_charged(t)))
        else:
            cross_section_model_neutral.form_factor.charged_variant = is_charged
            results.append(abs(cross_section_model_neutral(t)))

    return results


def function_nucleon_cross_section(
        ts: List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        nucleon_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: ModelParameters,
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)
    if _is_kaon_type_model(ff_model):
        raise ValueError('Received kaon type parameters!')
    if isinstance(ff_model, (ETGMRModel, TwoPolesModel)):
        raise ValueError('Received ETGMR or TwoPoles parameters!')

    config = ConfigParser()
    config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}

    cross_section_model = NucleonPairToElectronPositronTotalCrossSection(
        nucleon_mass, ff_model, config
    )

    results = []
    for datapoint in ts:
        t, is_proton, _ = _read_datapoint_nucleon(datapoint)  # type: ignore
        cross_section_model.form_factor.proton = is_proton
        results.append(abs(cross_section_model(t)))

    return results


def function_etgrm_or_twopoles_cross_section(
        ts:  Union[
            List[Union[KaonDatapoint, Tuple[float, float, float]]],
            List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        ],
        parameters: ModelParameters,
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    if isinstance(ff_model, (ETGMRModel, TwoPolesModel)):
        # In these cases the model can describe (with suitable parameters)
        # both form factors and cross-sections
        cross_section_model = ff_model
    else:
        raise ValueError(f'Unexpected parameter type: {type(parameters)}')

    results = []
    for datapoint in ts:
        if isinstance(datapoint, (KaonDatapoint, NucleonDatapoint)):
            t = datapoint.t
        else:
            t = float(datapoint[0])
        results.append(abs(cross_section_model(t)))

    return results


def function_kaon_form_factor_or_cross_section(
        xs: Union[List[Union[KaonDatapoint, Tuple[float, float, float]]]],
        charged_kaon_mass: float,
        neutral_kaon_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: ModelParameters,
        ) -> List[float]:

    if not _are_kaon_parameters(parameters):
        raise ValueError(f'Expected kaon parameters, got {type(parameters)}!')

    ff_model = _get_ff_model(parameters)
    config = ConfigParser()
    config['constants'] = {'alpha': str(alpha), 'hc_squared': str(hc_squared)}

    cs_model_charged = ScalarMesonProductionTotalCrossSection(
        charged_kaon_mass, ff_model, config)
    cs_model_neutral = ScalarMesonProductionTotalCrossSection(
        neutral_kaon_mass, ff_model, config)

    results = []
    for datapoint in xs:
        t, is_charged, is_for_cross_section = _read_datapoint_kaon(datapoint)
        if is_for_cross_section and is_charged:
            cs_model_charged.form_factor.charged_variant = is_charged
            results.append(abs(cs_model_charged(t)))
        elif is_for_cross_section and not is_charged:
            cs_model_neutral.form_factor.charged_variant = is_charged
            results.append(abs(cs_model_neutral(t)))
        else:
            ff_model.charged_variant = is_charged
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
        alpha: float, hc_squared: float,
        parameters: ModelParameters,
        nucleon_mass: Optional[float] = None,
        charged_kaon_mass: Optional[float] = None,
        neutral_kaon_mass: Optional[float] = None,
) -> Callable:
    parameters = parameters.copy()

    if _are_kaon_parameters(parameters):
        assert charged_kaon_mass is not None
        assert neutral_kaon_mass is not None

        def partial_f(ts, *args):
            parameters.update_free_values(list(args))
            return function_kaon_cross_section(ts, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, parameters)
    elif isinstance(parameters, NucleonParameters):
        assert nucleon_mass is not None

        def partial_f(ts, *args):
            parameters.update_free_values(list(args))
            return function_nucleon_cross_section(ts, nucleon_mass, alpha, hc_squared, parameters)
    elif isinstance(parameters, (ETGMRModelParameters, TwoPolesModelParameters)):
        def partial_f(ts, *args):
            parameters.update_free_values(list(args))
            return function_etgrm_or_twopoles_cross_section(ts, parameters)
    else:
        raise ValueError(f'Unexpected type of parameters: {type(parameters)}!')

    return partial_f


def make_partial_ff_or_cs_for_parameters(
        alpha: float, hc_squared: float, parameters: ModelParameters,
        charged_kaon_mass: Optional[float] = None,
        neutral_kaon_mass: Optional[float] = None
) -> Callable:
    parameters = parameters.copy()

    # TODO: implement other cases
    if not _are_kaon_parameters(parameters):
        raise NotImplementedError

    assert charged_kaon_mass is not None
    assert neutral_kaon_mass is not None

    def partial_f(ts, *args):
        parameters.update_free_values(list(args))
        return function_kaon_form_factor_or_cross_section(ts, charged_kaon_mass, neutral_kaon_mass,
                                                          alpha, hc_squared, parameters)

    return partial_f
