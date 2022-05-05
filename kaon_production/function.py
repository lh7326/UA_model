from configparser import ConfigParser
from typing import List, Tuple, Union

from ua_model.KaonUAModel import KaonUAModel
from ua_model.KaonUAModelSimplified import KaonUAModelSimplified
from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedRhoOmega
from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from kaon_production.data import Datapoint


def function_cross_section(
        ts: List[Union[Datapoint, Tuple[float, float]]],
        k_meson_mass: float,
        alpha: float,
        hc_squared: float,
        parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedRhoOmega],
        ) -> List[complex]:

    if isinstance(parameters, KaonParameters):
        ff_model = KaonUAModel(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersSimplified):
        ff_model = KaonUAModelSimplified(charged_variant=True, **{p.name: p.value for p in parameters})
    elif isinstance(parameters, KaonParametersFixedRhoOmega):
        ff_model = KaonUAModel(charged_variant=True, **{p.name: p.value for p in parameters})
    else:
        raise TypeError('Unexpected parameters type: ' + type(parameters).__name__)

    config = ConfigParser()
    config['constants'] = {'alpha': alpha, 'hc_squared': hc_squared}
    cross_section_model = ScalarMesonProductionTotalCrossSection(k_meson_mass, ff_model, config)

    results = []
    for datapoint in ts:
        if isinstance(datapoint, Datapoint):
            cross_section_model.form_factor.charged_variant = datapoint.is_charged
            results.append(cross_section_model(datapoint.t))
        else:
            cross_section_model.form_factor.charged_variant = bool(datapoint[1])
            results.append(cross_section_model(datapoint[0]))
    return results
