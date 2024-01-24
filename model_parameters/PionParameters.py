import math
import numpy as np
from typing import Dict, List, Tuple

from model_parameters.ModelParameters import Parameter, ModelParameters


class PionParameters(ModelParameters):

    def __init__(self,
                 t_0_isovector: float,
                 t_in_isovector: float,
                 mass_rho: float,
                 decay_rate_rho: float,
                 a_rho_prime: float,
                 mass_rho_prime: float,
                 decay_rate_rho_prime: float,
                 a_rho_double_prime: float,
                 mass_rho_double_prime: float,
                 decay_rate_rho_double_prime: float,
                 a_rho_triple_prime: float,
                 mass_rho_triple_prime: float,
                 decay_rate_rho_triple_prime: float,
                 w_pole: complex,
                 w_zero: complex,
                 ) -> None:

        super().__init__(
            t_0_isovector, t_in_isovector,
            mass_rho, decay_rate_rho,
            a_rho_prime, mass_rho_prime, decay_rate_rho_prime,
            a_rho_double_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
            a_rho_triple_prime, mass_rho_triple_prime, decay_rate_rho_triple_prime,
            w_pole, w_zero,
            always_fixed=('t_0_isovector',),
        )

    def _setup_data(self,
                    t_0_isovector: float,
                    t_in_isovector: float,
                    mass_rho: float,
                    decay_rate_rho: float,
                    a_rho_prime: float,
                    mass_rho_prime: float,
                    decay_rate_rho_prime: float,
                    a_rho_double_prime: float,
                    mass_rho_double_prime: float,
                    decay_rate_rho_double_prime: float,
                    a_rho_triple_prime: float,
                    mass_rho_triple_prime: float,
                    decay_rate_rho_triple_prime: float,
                    w_pole: complex,
                    w_zero: complex) -> List[Parameter]:

        return [
            Parameter(name='t_0_isovector', value=t_0_isovector, is_fixed=True),
            Parameter(name='t_in_isovector', value=t_in_isovector, is_fixed=False),
            Parameter(name='mass_rho', value=mass_rho, is_fixed=False),
            Parameter(name='decay_rate_rho', value=decay_rate_rho, is_fixed=False),
            Parameter(name='a_rho_prime', value=a_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, is_fixed=False),
            Parameter(name='a_rho_double_prime', value=a_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, is_fixed=False),
            Parameter(name='a_rho_triple_prime', value=a_rho_triple_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_triple_prime', value=mass_rho_triple_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_triple_prime', value=decay_rate_rho_triple_prime, is_fixed=False),
            Parameter(name='w_pole', value=w_pole, is_fixed=False),
            Parameter(name='w_zero', value=w_zero, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'PionParameters':
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

    def get_model_parameters_bounds_handpicked(self) -> Dict:
        """
        Returns a handpicked set of bounds.

        """
        return {
            't_in_isovector': {'lower': self['t_0_isovector'].value, 'upper': np.inf},
            'mass_rho': {'lower': 0.75, 'upper': 0.78},
            'decay_rate_rho': {'lower': 0.140, 'upper': 0.150},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': 1.28, 'upper': 1.48},
            'decay_rate_rho_prime': {'lower': 0.15, 'upper': 0.55},
            'a_rho_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': 1.68, 'upper': 1.74},
            'decay_rate_rho_double_prime': {'lower': 0.1, 'upper': 0.6},
            'a_rho_triple_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_triple_prime': {'lower': 2.0, 'upper': 2.4},
            'decay_rate_rho_triple_prime': {'lower': 0.1, 'upper': 0.6},
            'w_pole': {'lower': -np.inf, 'upper': np.inf},
            'w_zero': {'lower': -np.inf, 'upper': np.inf},
        }

    def get_model_parameters_bounds_maximal(self) -> Dict:
        """
        There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
        must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
        Decay rates must be non-negative and squared masses must lie above their respective t_0 treshold.

        """
        lower_mass_bound_isovector = math.sqrt(self['t_0_isovector'].value)
        return {
            't_in_isovector': {'lower': self['t_0_isovector'].value, 'upper': np.inf},
            'mass_rho': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho': {'lower': 0.0, 'upper': np.inf},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho_triple_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_triple_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_triple_prime': {'lower': 0.0, 'upper': np.inf},
            'w_pole': {'lower': -np.inf, 'upper': np.inf},
            'w_zero': {'lower': -np.inf, 'upper': np.inf},
        }

    def get_ordered_values(self) -> List[float]:
        """
        Print the values in the order:
          t_in_isovector
          mass_rho
          mass_rho_prime
          mass_rho_double_prime
          mass_rho_triple_prime
          decay_rate_rho
          decay_rate_rho_prime
          decay_rate_rho_double_prime
          decay_rate_rho_triple_prime
          a_rho_prime
          a_rho_double_prime
          a_rho_triple_prime
          w_pole
          w_zero

        Returns: the ordered list of parameter values

        """
        names = [
            't_in_isovector',
            'mass_rho',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'mass_rho_triple_prime',
            'decay_rate_rho',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'decay_rate_rho_triple_prime',
            'a_rho_prime',
            'a_rho_double_prime',
            'a_rho_triple_prime',
            'w_pole',
            'w_zero',
        ]
        return [self[name].value for name in names]

    @classmethod
    def from_ordered_values(
            cls,
            list_of_values: List[float],
            t_0_isovector: float) -> 'PionParameters':
        names = [
            't_in_isovector',
            'mass_rho',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'mass_rho_triple_prime',
            'decay_rate_rho',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'decay_rate_rho_triple_prime',
            'a_rho_prime',
            'a_rho_double_prime',
            'a_rho_triple_prime',
            'w_pole',
            'w_zero',
        ]
        assert len(list_of_values) == len(names)
        pars = [Parameter(name, value, False) for name, value in zip(names, list_of_values)]
        pars.extend([
            Parameter('t_0_isovector', t_0_isovector, True),
        ])
        return cls.from_list(pars)

    def fix_resonances(self):
        self.fix_parameters(['mass_rho', 'mass_rho_prime', 'mass_rho_double_prime', 'mass_rho_triple_prime',
                             'decay_rate_rho', 'decay_rate_rho_prime', 'decay_rate_rho_double_prime',
                             'decay_rate_rho_triple_prime'])
