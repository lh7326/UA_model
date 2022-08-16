from configparser import ConfigParser
from typing import List, Tuple, Union

from ua_model.NucleonUAModel import NucleonUAModel
from other_models import ETGMRModel, TwoPolesModel
from model_parameters import ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters
from nucleon_production.data import NucleonDatapoint


def _get_ff_model(
        parameters: Union[ETGMRModelParameters, TwoPolesModelParameters, NucleonParameters],
) -> Union[ETGMRModel, TwoPolesModel, NucleonUAModel]:
    if isinstance(parameters, ETGMRModelParameters):
        return ETGMRModel(a=parameters['a'].value, m_a=parameters['m_a'].value)
    elif isinstance(parameters, TwoPolesModelParameters):
        return TwoPolesModel(a=parameters['a'].value, m_1=parameters['m_1'].value, m_2=parameters['m_2'].value)
    elif isinstance(parameters, NucleonParameters):
        return NucleonUAModel(proton=True, electric=True, **{p.name: p.value for p in parameters})
    else:
        raise TypeError('Unexpected parameters type: ' + type(parameters).__name__)


def _read_datapoint_nucleon(
        datapoint: Union[NucleonDatapoint, Tuple[complex, float, float]]
) -> Tuple[complex, bool, bool]:
    if isinstance(datapoint, NucleonDatapoint):
        return datapoint.t, datapoint.proton, datapoint.electric
    else:
        return complex(datapoint[0]), bool(datapoint[1]), bool(datapoint[2])


def function_nucleon_cross_section(
        ts: List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        ) -> List[complex]:

    raise 'Not Implemented'


def function_nucleon_form_factor(
        ts: List[Union[NucleonDatapoint, Tuple[float, float, float]]],
        parameters: NucleonParameters,
        ) -> List[complex]:

    ff_model = _get_ff_model(parameters)

    results = []
    for datapoint in ts:
        t, proton, electric = _read_datapoint_nucleon(datapoint)
        ff_model.proton = proton
        ff_model.electric = electric
        results.append(abs(ff_model(t)))

    return results
