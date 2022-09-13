from configparser import ConfigParser
from typing import List, Tuple, Union

from ua_model.KaonUAModel import KaonUAModel
from ua_model.KaonUAModelSimplified import KaonUAModelSimplified
from ua_model.KaonUAModelB import KaonUAModelB
from ua_model.NucleonUAModel import NucleonUAModel
from other_models import ETGMRModel, TwoPolesModel
from model_parameters import (KaonParameters, KaonParametersB, KaonParametersSimplified,
                              KaonParametersFixedRhoOmega, KaonParametersFixedSelected,
                              ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters)
from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from kaon_production.data import Datapoint


def _get_ff_model(
        parameters: Union[KaonParameters, KaonParametersB, KaonParametersSimplified,
                          KaonParametersFixedRhoOmega, KaonParametersFixedSelected, NucleonParameters],
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
    elif isinstance(parameters, ETGMRModelParameters):
        return ETGMRModel(a=parameters['a'].value, m_a=parameters['m_a'].value, m_d=parameters['m_d'].value)
    elif isinstance(parameters, TwoPolesModelParameters):
        return TwoPolesModel(a=parameters['a'].value, m_1=parameters['m_1'].value, m_2=parameters['m_2'].value)
    else:
        raise TypeError('Unexpected parameters type: ' + type(parameters).__name__)


def _read_datapoint_kaon(datapoint: Union[Datapoint, Tuple[complex, float]]) -> Tuple[complex, bool]:
    if isinstance(datapoint, Datapoint):
        return datapoint.t, datapoint.is_charged
    else:
        return complex(datapoint[0]), bool(datapoint[1])


def function_kaon_cross_section(
        ts: List[Union[Datapoint, Tuple[float, float]]],
        k_meson_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: Union[
            KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedRhoOmega,
            KaonParametersFixedSelected],
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    config = ConfigParser()
    config['constants'] = {'alpha': alpha, 'hc_squared': hc_squared}
    cross_section_model = ScalarMesonProductionTotalCrossSection(k_meson_mass, ff_model, config)

    results = []
    for datapoint in ts:
        t, is_charged = _read_datapoint_kaon(datapoint)
        cross_section_model.form_factor.charged_variant = is_charged
        results.append(cross_section_model(t))

    return results


def function_kaon_form_factor(
        ts: List[Union[Datapoint, Tuple[float, float]]],
        parameters: Union[
            KaonParameters, KaonParametersB, KaonParametersFixedRhoOmega, KaonParametersFixedSelected,
        ],
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    results = []
    for datapoint in ts:
        t, is_charged = _read_datapoint_kaon(datapoint)
        ff_model.charged_variant = is_charged
        results.append(abs(ff_model(t)))

    return results
