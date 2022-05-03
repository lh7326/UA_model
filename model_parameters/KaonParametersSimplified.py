import math
import numpy as np
from typing import Dict, List, Tuple

from model_parameters.ModelParameters import Parameter, ModelParameters


class KaonParametersSimplified(ModelParameters):

    def __init__(self,
                 t_0_isoscalar: float,
                 t_0_isovector: float,
                 t_in_isoscalar: float,
                 t_in_isovector: float,
                 a_omega_prime: float,
                 mass_omega_prime: float,
                 decay_rate_omega_prime: float,
                 a_omega_double_prime: float,
                 mass_omega_double_prime: float,
                 decay_rate_omega_double_prime: float,
                 a_phi: float,
                 mass_phi: float,
                 decay_rate_phi: float,
                 a_phi_prime: float,
                 mass_phi_prime: float,
                 decay_rate_phi_prime: float,
                 mass_phi_double_prime: float,
                 decay_rate_phi_double_prime: float,
                 a_rho_prime: float,
                 mass_rho_prime: float,
                 decay_rate_rho_prime: float,
                 a_rho_double_prime: float,
                 mass_rho_double_prime: float,
                 decay_rate_rho_double_prime: float,
                 mass_rho_triple_prime: float,
                 decay_rate_rho_triple_prime: float) -> None:

        super().__init__(
            t_0_isoscalar, t_0_isovector, t_in_isoscalar, t_in_isovector,
            a_omega_prime, mass_omega_prime, decay_rate_omega_prime,
            a_omega_double_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
            a_phi, mass_phi, decay_rate_phi, a_phi_prime, mass_phi_prime, decay_rate_phi_prime,
            mass_phi_double_prime, decay_rate_phi_double_prime,
            a_rho_prime, mass_rho_prime, decay_rate_rho_prime,
            a_rho_double_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
            mass_rho_triple_prime, decay_rate_rho_triple_prime,
            always_fixed=('t_0_isoscalar', 't_0_isovector'),
        )

    def _setup_data(self,
                    t_0_isoscalar: float,
                    t_0_isovector: float,
                    t_in_isoscalar: float,
                    t_in_isovector: float,
                    a_omega_prime: float,
                    mass_omega_prime: float,
                    decay_rate_omega_prime: float,
                    a_omega_double_prime: float,
                    mass_omega_double_prime: float,
                    decay_rate_omega_double_prime: float,
                    a_phi: float,
                    mass_phi: float,
                    decay_rate_phi: float,
                    a_phi_prime: float,
                    mass_phi_prime: float,
                    decay_rate_phi_prime: float,
                    mass_phi_double_prime: float,
                    decay_rate_phi_double_prime: float,
                    a_rho_prime: float,
                    mass_rho_prime: float,
                    decay_rate_rho_prime: float,
                    a_rho_double_prime: float,
                    mass_rho_double_prime: float,
                    decay_rate_rho_double_prime: float,
                    mass_rho_triple_prime: float,
                    decay_rate_rho_triple_prime: float) -> List[Parameter]:

        return [
            Parameter(name='t_0_isoscalar', value=t_0_isoscalar, is_fixed=True),
            Parameter(name='t_0_isovector', value=t_0_isovector, is_fixed=True),
            Parameter(name='t_in_isoscalar', value=t_in_isoscalar, is_fixed=False),
            Parameter(name='t_in_isovector', value=t_in_isovector, is_fixed=False),
            Parameter(name='a_omega_prime', value=a_omega_prime, is_fixed=False),
            Parameter(name='mass_omega_prime', value=mass_omega_prime, is_fixed=False),
            Parameter(name='decay_rate_omega_prime', value=decay_rate_omega_prime, is_fixed=False),
            Parameter(name='a_omega_double_prime', value=a_omega_double_prime, is_fixed=False),
            Parameter(name='mass_omega_double_prime', value=mass_omega_double_prime, is_fixed=False),
            Parameter(name='decay_rate_omega_double_prime', value=decay_rate_omega_double_prime, is_fixed=False),
            Parameter(name='a_phi', value=a_phi, is_fixed=False),
            Parameter(name='mass_phi', value=mass_phi, is_fixed=False),
            Parameter(name='decay_rate_phi', value=decay_rate_phi, is_fixed=False),
            Parameter(name='a_phi_prime', value=a_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_prime', value=mass_phi_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_prime', value=decay_rate_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_double_prime', value=mass_phi_double_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_double_prime', value=decay_rate_phi_double_prime, is_fixed=False),
            Parameter(name='a_rho_prime', value=a_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, is_fixed=False),
            Parameter(name='a_rho_double_prime', value=a_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_triple_prime', value=mass_rho_triple_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_triple_prime', value=decay_rate_rho_triple_prime, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'KaonParametersSimplified':
        kwargs = {par.name: par.value for par in list_of_parameters}
        return cls(**kwargs)

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
            't_in_isoscalar': {'lower': self['t_0_isoscalar'].value, 'upper': np.inf},
            't_in_isovector': {'lower': self['t_0_isovector'].value, 'upper': np.inf},
            'a_omega_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_prime': {'lower': 1.37, 'upper': 1.45},
            'decay_rate_omega_prime': {'lower': 0.25, 'upper': 0.33},
            'a_omega_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_double_prime': {'lower': 1.665, 'upper': 1.675},
            'decay_rate_omega_double_prime': {'lower': 0.30, 'upper': 0.33},
            'a_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': 1.0, 'upper': 1.05},
            'decay_rate_phi': {'lower': 0.0040, 'upper': 0.0045},
            'a_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': 1.675, 'upper': 1.685},
            'decay_rate_phi_prime': {'lower': 0.1, 'upper': 0.2},
            'mass_phi_double_prime': {'lower': 2.1, 'upper': 2.2},
            'decay_rate_phi_double_prime': {'lower': 0.05, 'upper': 0.2},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': 1.28, 'upper': 1.48},
            'decay_rate_rho_prime': {'lower': 0.15, 'upper': 0.55},
            'a_rho_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': 1.68, 'upper': 1.74},
            'decay_rate_rho_double_prime': {'lower': 0.1, 'upper': 0.6},
            'mass_rho_triple_prime': {'lower': 2.0, 'upper': 2.4},
            'decay_rate_rho_triple_prime': {'lower': 0.1, 'upper': 0.6},
        }

    def get_model_parameters_bounds_maximal(self) -> Dict:
        """
        There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
        must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
        Decay rates must be non-negative and squared masses must lie above their respective t_0 treshold.

        """
        lower_mass_bound_isoscalar = math.sqrt(self['t_0_isoscalar'].value)
        lower_mass_bound_isovector = math.sqrt(self['t_0_isovector'].value)
        return {
            't_in_isoscalar': {'lower': self['t_0_isoscalar'].value, 'upper': np.inf},
            't_in_isovector': {'lower': self['t_0_isovector'].value, 'upper': np.inf},
            'a_omega_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega_prime': {'lower': 0.0, 'upper': np.inf},
            'a_omega_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_double_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi': {'lower': 0.0, 'upper': np.inf},
            'a_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_phi_double_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_double_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_rho_triple_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_triple_prime': {'lower': 0.0, 'upper': np.inf},
        }

    def get_ordered_values(self) -> List[float]:
        """
        Print the values in the order:
          t_in_isoscalar
          t_in_isovector
          mass_rho_prime
          mass_rho_double_prime
          mass_rho_triple_prime
          decay_rate_rho_prime
          decay_rate_rho_double_prime
          decay_rate_rho_triple_prime
          a_rho_prime
          a_rho_double_prime
          mass_phi
          mass_omega_prime
          mass_omega_double_prime
          mass_phi_prime
          mass_phi_double_prime
          decay_rate_phi
          decay_rate_omega_prime
          decay_rate_omega_double_prime
          decay_rate_phi_prime
          decay_rate_phi_double_prime
          a_phi
          a_omega_prime
          a_omega_double_prime
          a_phi_prime

        Returns: the ordered list of parameter values

        """
        names = [
            't_in_isoscalar',
            't_in_isovector',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'mass_rho_triple_prime',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'decay_rate_rho_triple_prime',
            'a_rho_prime',
            'a_rho_double_prime',
            'mass_phi',
            'mass_omega_prime',
            'mass_omega_double_prime',
            'mass_phi_prime',
            'mass_phi_double_prime',
            'decay_rate_phi',
            'decay_rate_omega_prime',
            'decay_rate_omega_double_prime',
            'decay_rate_phi_prime',
            'decay_rate_phi_double_prime',
            'a_phi',
            'a_omega_prime',
            'a_omega_double_prime',
            'a_phi_prime',
        ]
        return [self[name].value for name in names]

    @classmethod
    def from_ordered_values(
            cls,
            list_of_values: List[float],
            t_0_isoscalar: float,
            t_0_isovector: float) -> 'KaonParametersSimplified':
        names = [
            't_in_isoscalar',
            't_in_isovector',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'mass_rho_triple_prime',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'decay_rate_rho_triple_prime',
            'a_rho_prime',
            'a_rho_double_prime',
            'mass_phi',
            'mass_omega_prime',
            'mass_omega_double_prime',
            'mass_phi_prime',
            'mass_phi_double_prime',
            'decay_rate_phi',
            'decay_rate_omega_prime',
            'decay_rate_omega_double_prime',
            'decay_rate_phi_prime',
            'decay_rate_phi_double_prime',
            'a_phi',
            'a_omega_prime',
            'a_omega_double_prime',
            'a_phi_prime',
        ]
        assert len(list_of_values) == len(names)
        pars = [Parameter(name, value, False) for name, value in zip(names, list_of_values)]
        pars.extend([
            Parameter('t_0_isoscalar', t_0_isoscalar, True),
            Parameter('t_0_isovector', t_0_isovector, True),
        ])
        return cls.from_list(pars)

    def fix_resonances(self):
        self.fix_parameters(['mass_rho_prime', 'mass_rho_double_prime', 'mass_rho_triple_prime',
                             'decay_rate_rho_prime', 'decay_rate_rho_double_prime', 'decay_rate_rho_triple_prime',
                             'mass_phi', 'mass_omega_prime', 'mass_omega_double_prime',
                             'mass_phi_prime', 'mass_phi_double_prime', 'decay_rate_phi',
                             'decay_rate_omega_prime', 'decay_rate_omega_double_prime',
                             'decay_rate_phi_prime', 'decay_rate_phi_double_prime',
                             ])
