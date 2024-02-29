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
                 a_omega: float,
                 mass_omega: float,
                 decay_rate_omega: float,
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
                 a_rho: float,
                 mass_rho: float,
                 decay_rate_rho: float,
                 a_rho_prime: float,
                 mass_rho_prime: float,
                 decay_rate_rho_prime: float,
                 mass_rho_double_prime: float,
                 decay_rate_rho_double_prime: float) -> None:

        always_fixed = (
            't_0_isoscalar', 't_0_isovector',
            'mass_omega', 'decay_rate_omega',
            'mass_omega_double_prime',
            'mass_rho', 'decay_rate_rho',
        )
        super().__init__(
            t_0_isoscalar, t_0_isovector, t_in_isoscalar, t_in_isovector,
            a_omega, mass_omega, decay_rate_omega,
            a_omega_double_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
            a_phi, mass_phi, decay_rate_phi, a_phi_prime, mass_phi_prime, decay_rate_phi_prime,
            mass_phi_double_prime, decay_rate_phi_double_prime,
            a_rho, mass_rho, decay_rate_rho, a_rho_prime, mass_rho_prime, decay_rate_rho_prime,
            mass_rho_double_prime, decay_rate_rho_double_prime,
            always_fixed=always_fixed,
        )

    def _setup_data(self,
                    t_0_isoscalar: float,
                    t_0_isovector: float,
                    t_in_isoscalar: float,
                    t_in_isovector: float,
                    a_omega: float,
                    mass_omega: float,
                    decay_rate_omega: float,
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
                    a_rho: float,
                    mass_rho: float,
                    decay_rate_rho: float,
                    a_rho_prime: float,
                    mass_rho_prime: float,
                    decay_rate_rho_prime: float,
                    mass_rho_double_prime: float,
                    decay_rate_rho_double_prime: float) -> List[Parameter]:

        return [
            Parameter(name='t_0_isoscalar', value=t_0_isoscalar, is_fixed=True),
            Parameter(name='t_0_isovector', value=t_0_isovector, is_fixed=True),
            Parameter(name='t_in_isoscalar', value=t_in_isoscalar, is_fixed=False),
            Parameter(name='t_in_isovector', value=t_in_isovector, is_fixed=False),
            Parameter(name='a_omega', value=a_omega, is_fixed=False),
            Parameter(name='mass_omega', value=mass_omega, is_fixed=True),
            Parameter(name='decay_rate_omega', value=decay_rate_omega, is_fixed=True),
            Parameter(name='a_omega_double_prime', value=a_omega_double_prime, is_fixed=False),
            Parameter(name='mass_omega_double_prime', value=mass_omega_double_prime, is_fixed=True),
            Parameter(name='decay_rate_omega_double_prime', value=decay_rate_omega_double_prime, is_fixed=False),
            Parameter(name='a_phi', value=a_phi, is_fixed=False),
            Parameter(name='mass_phi', value=mass_phi, is_fixed=False),
            Parameter(name='decay_rate_phi', value=decay_rate_phi, is_fixed=False),
            Parameter(name='a_phi_prime', value=a_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_prime', value=mass_phi_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_prime', value=decay_rate_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_double_prime', value=mass_phi_double_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_double_prime', value=decay_rate_phi_double_prime, is_fixed=False),
            Parameter(name='a_rho', value=a_rho, is_fixed=False),
            Parameter(name='mass_rho', value=mass_rho, is_fixed=True),
            Parameter(name='decay_rate_rho', value=decay_rate_rho, is_fixed=True),
            Parameter(name='a_rho_prime', value=a_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'KaonParametersSimplified':
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
            't_in_isoscalar': {'lower': self['t_0_isoscalar'].value, 'upper': np.inf},
            't_in_isovector': {'lower': self['t_0_isovector'].value, 'upper': np.inf},
            'a_omega': {'lower': -np.inf, 'upper': np.inf},
            'a_omega_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'decay_rate_omega_double_prime': {'lower': 0.2, 'upper': 0.4},
            'a_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': 1.0, 'upper': 1.1},
            'decay_rate_phi': {'lower': 0.003, 'upper': 0.005},
            'a_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': 1.6, 'upper': 1.7},
            'decay_rate_phi_prime': {'lower': 0.1, 'upper': 0.2},
            'mass_phi_double_prime': {'lower': 2.0, 'upper': 2.3},
            'decay_rate_phi_double_prime': {'lower': 0.05, 'upper': 0.3},
            'a_rho': {'lower': -np.inf, 'upper': np.inf},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': 1.2, 'upper': 1.5},
            'decay_rate_rho_prime': {'lower': 0.15, 'upper': 0.55},
            'mass_rho_double_prime': {'lower': 1.6, 'upper': 1.9},
            'decay_rate_rho_double_prime': {'lower': 0.1, 'upper': 0.6},
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
            'a_omega': {'lower': -np.inf, 'upper': np.inf},
            'a_omega_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'decay_rate_omega_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi': {'lower': 0.0, 'upper': np.inf},
            'a_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_phi_double_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho': {'lower': -np.inf, 'upper': np.inf},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_double_prime': {'lower': 0.0, 'upper': np.inf},
        }

    def get_ordered_values(self) -> List[float]:
        """
        Print the values in the order:
            't_in_isoscalar',
            't_in_isovector',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'a_rho',
            'a_rho_prime',
            'mass_phi',
            'mass_omega_double_prime',
            'mass_phi_prime',
            'mass_phi_double_prime',
            'decay_rate_phi',
            'decay_rate_omega_double_prime',
            'decay_rate_phi_prime',
            'decay_rate_phi_double_prime',
            'a_phi',
            'a_omega_double_prime',
            'a_phi_prime',

        Returns: the ordered list of parameter values

        """
        names = [
            't_in_isoscalar',
            't_in_isovector',
            'mass_rho_prime',
            'mass_rho_double_prime',
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'a_rho',
            'a_rho_prime',
            'mass_phi',
            'mass_omega_double_prime',
            'mass_phi_prime',
            'mass_phi_double_prime',
            'decay_rate_phi',
            'decay_rate_omega_double_prime',
            'decay_rate_phi_prime',
            'decay_rate_phi_double_prime',
            'a_phi',
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
            'decay_rate_rho_prime',
            'decay_rate_rho_double_prime',
            'a_rho',
            'a_rho_prime',
            'mass_phi',
            'mass_omega_double_prime',
            'mass_phi_prime',
            'mass_phi_double_prime',
            'decay_rate_phi',
            'decay_rate_omega_double_prime',
            'decay_rate_phi_prime',
            'decay_rate_phi_double_prime',
            'a_phi',
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
        self.fix_parameters(['mass_rho', 'mass_rho_prime', 'mass_rho_double_prime',
                             'decay_rate_rho', 'decay_rate_rho_prime', 'decay_rate_rho_double_prime',
                             'mass_omega', 'mass_phi', 'mass_omega_double_prime', 'mass_phi_prime',
                             'mass_phi_double_prime', 'decay_rate_omega', 'decay_rate_phi', 'decay_rate_omega_double_prime',
                             'decay_rate_phi_prime','decay_rate_phi_double_prime',
                             ])
