import math
import numpy as np
from typing import Dict, List, Tuple

from model_parameters.ModelParameters import Parameter, ModelParameters


class NucleonParameters(ModelParameters):

    def __init__(self,
                 t_0_dirac_isoscalar: float,
                 t_0_dirac_isovector: float,
                 t_in_dirac_isoscalar: float,
                 t_in_dirac_isovector: float,
                 t_0_pauli_isoscalar: float,
                 t_0_pauli_isovector: float,
                 t_in_pauli_isoscalar: float,
                 t_in_pauli_isovector: float,
                 a_dirac_omega: float,
                 a_pauli_omega: float,
                 mass_omega: float,
                 decay_rate_omega: float,
                 a_dirac_omega_prime: float,
                 mass_omega_prime: float,
                 decay_rate_omega_prime: float,
                 mass_omega_double_prime: float,
                 decay_rate_omega_double_prime: float,
                 a_dirac_phi: float,
                 a_pauli_phi: float,
                 mass_phi: float,
                 decay_rate_phi: float,
                 a_dirac_phi_prime: float,
                 a_pauli_phi_prime: float,
                 mass_phi_prime: float,
                 decay_rate_phi_prime: float,
                 mass_phi_double_prime: float,
                 decay_rate_phi_double_prime: float,
                 a_dirac_rho: float,
                 mass_rho: float,
                 decay_rate_rho: float,
                 mass_rho_prime: float,
                 decay_rate_rho_prime: float,
                 mass_rho_double_prime: float,
                 decay_rate_rho_double_prime: float) -> None:

        super().__init__(
            t_0_dirac_isoscalar, t_0_dirac_isovector, t_in_dirac_isoscalar, t_in_dirac_isovector,
            t_0_pauli_isoscalar, t_0_pauli_isovector, t_in_pauli_isoscalar, t_in_pauli_isovector,
            a_dirac_omega, a_pauli_omega, mass_omega, decay_rate_omega, a_dirac_omega_prime,
            mass_omega_prime, decay_rate_omega_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
            a_dirac_phi, a_pauli_phi, mass_phi, decay_rate_phi, a_dirac_phi_prime, a_pauli_phi_prime,
            mass_phi_prime, decay_rate_phi_prime, mass_phi_double_prime, decay_rate_phi_double_prime,
            a_dirac_rho, mass_rho, decay_rate_rho, mass_rho_prime, decay_rate_rho_prime,
            mass_rho_double_prime, decay_rate_rho_double_prime,
            always_fixed=('t_0_dirac_isoscalar', 't_0_dirac_isovector', 't_0_pauli_isoscalar', 't_0_pauli_isovector'),
        )

    def _setup_data(self,
                    t_0_dirac_isoscalar: float,
                    t_0_dirac_isovector: float,
                    t_in_dirac_isoscalar: float,
                    t_in_dirac_isovector: float,
                    t_0_pauli_isoscalar: float,
                    t_0_pauli_isovector: float,
                    t_in_pauli_isoscalar: float,
                    t_in_pauli_isovector: float,
                    a_dirac_omega: float,
                    a_pauli_omega: float,
                    mass_omega: float,
                    decay_rate_omega: float,
                    a_dirac_omega_prime: float,
                    mass_omega_prime: float,
                    decay_rate_omega_prime: float,
                    mass_omega_double_prime: float,
                    decay_rate_omega_double_prime: float,
                    a_dirac_phi: float,
                    a_pauli_phi: float,
                    mass_phi: float,
                    decay_rate_phi: float,
                    a_dirac_phi_prime: float,
                    a_pauli_phi_prime: float,
                    mass_phi_prime: float,
                    decay_rate_phi_prime: float,
                    mass_phi_double_prime: float,
                    decay_rate_phi_double_prime: float,
                    a_dirac_rho: float,
                    mass_rho: float,
                    decay_rate_rho: float,
                    mass_rho_prime: float,
                    decay_rate_rho_prime: float,
                    mass_rho_double_prime: float,
                    decay_rate_rho_double_prime: float) -> List[Parameter]:

        return [
            Parameter(name='t_0_dirac_isoscalar', value=t_0_dirac_isoscalar, is_fixed=True),
            Parameter(name='t_0_dirac_isovector', value=t_0_dirac_isovector, is_fixed=True),
            Parameter(name='t_in_dirac_isoscalar', value=t_in_dirac_isoscalar, is_fixed=False),
            Parameter(name='t_in_dirac_isovector', value=t_in_dirac_isovector, is_fixed=False),
            Parameter(name='t_0_pauli_isoscalar', value=t_0_pauli_isoscalar, is_fixed=True),
            Parameter(name='t_0_pauli_isovector', value=t_0_pauli_isovector, is_fixed=True),
            Parameter(name='t_in_pauli_isoscalar', value=t_in_pauli_isoscalar, is_fixed=False),
            Parameter(name='t_in_pauli_isovector', value=t_in_pauli_isovector, is_fixed=False),
            Parameter(name='a_dirac_omega', value=a_dirac_omega, is_fixed=False),
            Parameter(name='a_pauli_omega', value=a_pauli_omega, is_fixed=False),
            Parameter(name='mass_omega', value=mass_omega, is_fixed=False),
            Parameter(name='decay_rate_omega', value=decay_rate_omega, is_fixed=False),
            Parameter(name='a_dirac_omega_prime', value=a_dirac_omega_prime, is_fixed=False),
            Parameter(name='mass_omega_prime', value=mass_omega_prime, is_fixed=False),
            Parameter(name='decay_rate_omega_prime', value=decay_rate_omega_prime, is_fixed=False),
            Parameter(name='mass_omega_double_prime', value=mass_omega_double_prime, is_fixed=False),
            Parameter(name='decay_rate_omega_double_prime', value=decay_rate_omega_double_prime, is_fixed=False),
            Parameter(name='a_dirac_phi', value=a_dirac_phi, is_fixed=False),
            Parameter(name='a_pauli_phi', value=a_pauli_phi, is_fixed=False),
            Parameter(name='mass_phi', value=mass_phi, is_fixed=False),
            Parameter(name='decay_rate_phi', value=decay_rate_phi, is_fixed=False),
            Parameter(name='a_dirac_phi_prime', value=a_dirac_phi_prime, is_fixed=False),
            Parameter(name='a_pauli_phi_prime', value=a_pauli_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_prime', value=mass_phi_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_prime', value=decay_rate_phi_prime, is_fixed=False),
            Parameter(name='mass_phi_double_prime', value=mass_phi_double_prime, is_fixed=False),
            Parameter(name='decay_rate_phi_double_prime', value=decay_rate_phi_double_prime, is_fixed=False),
            Parameter(name='a_dirac_rho', value=a_dirac_rho, is_fixed=False),
            Parameter(name='mass_rho', value=mass_rho, is_fixed=False),
            Parameter(name='decay_rate_rho', value=decay_rate_rho, is_fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'NucleonParameters':
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
            't_in_dirac_isoscalar': {'lower': self['t_0_dirac_isoscalar'].value, 'upper': np.inf},
            't_in_dirac_isovector': {'lower': self['t_0_dirac_isovector'].value, 'upper': np.inf},
            't_in_pauli_isoscalar': {'lower': self['t_0_pauli_isoscalar'].value, 'upper': np.inf},
            't_in_pauli_isovector': {'lower': self['t_0_pauli_isovector'].value, 'upper': np.inf},
            'a_dirac_omega': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_omega': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega': {'lower': 0.775, 'upper': 0.79},
            'decay_rate_omega': {'lower': 0.0085, 'upper': 0.0089},
            'a_dirac_omega_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_prime': {'lower': 1.37, 'upper': 1.45},
            'decay_rate_omega_prime': {'lower': 0.25, 'upper': 0.33},
            'mass_omega_double_prime': {'lower': 1.665, 'upper': 1.675},
            'decay_rate_omega_double_prime': {'lower': 0.30, 'upper': 0.33},
            'a_dirac_phi': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': 1.0, 'upper': 1.05},
            'decay_rate_phi': {'lower': 0.0040, 'upper': 0.0045},
            'a_dirac_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': 1.675, 'upper': 1.685},
            'decay_rate_phi_prime': {'lower': 0.1, 'upper': 0.2},
            'mass_phi_double_prime': {'lower': 2.1, 'upper': 2.2},
            'decay_rate_phi_double_prime': {'lower': 0.05, 'upper': 0.2},
            'a_dirac_rho': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho': {'lower': 0.75, 'upper': 0.78},
            'decay_rate_rho': {'lower': 0.140, 'upper': 0.150},
            'mass_rho_prime': {'lower': 1.28, 'upper': 1.48},
            'decay_rate_rho_prime': {'lower': 0.15, 'upper': 0.55},
            'mass_rho_double_prime': {'lower': 1.68, 'upper': 1.74},
            'decay_rate_rho_double_prime': {'lower': 0.1, 'upper': 0.6},
        }

    def get_model_parameters_bounds_maximal(self) -> Dict:
        """
        There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
        must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
        Decay rates must be non-negative and squared masses must lie above their respective t_0 treshold.

        """
        lower_mass_bound_dirac_isoscalar = math.sqrt(self['t_0_dirac_isoscalar'].value)
        lower_mass_bound_dirac_isovector = math.sqrt(self['t_0_dirac_isovector'].value)
        lower_mass_bound_pauli_isoscalar = math.sqrt(self['t_0_pauli_isoscalar'].value)
        lower_mass_bound_pauli_isovector = math.sqrt(self['t_0_pauli_isovector'].value)
        lower_mass_bound_isoscalar = max(lower_mass_bound_dirac_isoscalar, lower_mass_bound_pauli_isoscalar)
        lower_mass_bound_isovector = max(lower_mass_bound_dirac_isovector, lower_mass_bound_pauli_isovector)
        return {
            't_in_dirac_isoscalar': {'lower': self['t_0_dirac_isoscalar'].value, 'upper': np.inf},
            't_in_dirac_isovector': {'lower': self['t_0_dirac_isovector'].value, 'upper': np.inf},
            't_in_pauli_isoscalar': {'lower': self['t_0_pauli_isoscalar'].value, 'upper': np.inf},
            't_in_pauli_isovector': {'lower': self['t_0_pauli_isovector'].value, 'upper': np.inf},
            'a_dirac_omega': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_omega': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega': {'lower': 0.0, 'upper': np.inf},
            'a_dirac_omega_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_omega_double_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_dirac_phi': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_phi': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi': {'lower': 0.0, 'upper': np.inf},
            'a_dirac_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'a_pauli_phi_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_phi_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_phi_double_prime': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_phi_double_prime': {'lower': 0.0, 'upper': np.inf},
            'a_dirac_rho': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho': {'lower': 0.0, 'upper': np.inf},
            'mass_rho_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_double_prime': {'lower': 0.0, 'upper': np.inf},
        }

    def fix_resonances(self):
        self.fix_parameters(['mass_rho', 'mass_rho_prime', 'mass_rho_double_prime',
                             'decay_rate_rho', 'decay_rate_rho_prime', 'decay_rate_rho_double_prime',
                             'mass_omega', 'mass_phi', 'mass_omega_prime', 'mass_omega_double_prime',
                             'mass_phi_prime', 'mass_phi_double_prime', 'decay_rate_omega',
                             'decay_rate_phi', 'decay_rate_omega_prime', 'decay_rate_omega_double_prime',
                             'decay_rate_phi_prime','decay_rate_phi_double_prime',
                             ])
