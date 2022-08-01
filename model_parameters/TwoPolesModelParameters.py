import numpy as np
from typing import Dict, List, Tuple

from model_parameters.ModelParameters import Parameter, ModelParameters


class TwoPolesModelParameters(ModelParameters):

    def __init__(self, a: float, m_1: float, m_2: float) -> None:

        super().__init__(a, m_1, m_2, always_fixed=())

    def _setup_data(self, a: float, m_1: float, m_2: float) -> List[Parameter]:

        return [
            Parameter(name='a', value=a, is_fixed=False),
            Parameter(name='m_1', value=m_1, is_fixed=False),
            Parameter(name='m_2', value=m_2, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'TwoPolesModelParameters':
        kwargs = {par.name: par.value for par in list_of_parameters}
        instance = cls(**kwargs)
        parameters_to_fix = [p.name for p in list_of_parameters if p.is_fixed]
        instance.release_all_parameters()
        instance.fix_parameters(parameters_to_fix)
        return instance

    def get_bounds_for_free_parameters(self, handpicked: bool = True) -> Tuple[List[float], List[float]]:
        if handpicked:
            full_bounds = self.get_model_parameters_bounds_handpicked()
        else:
            full_bounds = self.get_model_parameters_bounds_maximal()
        lower_bounds = []
        upper_bounds = []
        for parameter in self._data:
            if not parameter.is_fixed:
                bounds = full_bounds[parameter.name]
                lower_bounds.append(bounds['lower'])
                upper_bounds.append(bounds['upper'])
        return lower_bounds, upper_bounds

    @staticmethod
    def get_model_parameters_bounds_handpicked() -> Dict:
        """
        Returns a handpicked set of bounds.

        """
        return {
            'a': {'lower': -np.inf, 'upper': np.inf},
            'm_1': {'lower': 0.01, 'upper': np.inf},
            'm_2': {'lower': 0.1, 'upper': np.inf},
        }

    @staticmethod
    def get_model_parameters_bounds_maximal() -> Dict:
        return {
            'a': {'lower': -np.inf, 'upper': np.inf},
            'm_1': {'lower': 0.0, 'upper': np.inf},
            'm_2': {'lower': 0.0, 'upper': np.inf},
        }
