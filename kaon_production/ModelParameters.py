from collections import namedtuple
import math
import numpy as np


Parameter = namedtuple('Parameter', 'name value is_fixed')


class ModelParameters:

    def __init__(self,
                 t_0_isoscalar, t_0_isovector,
                 t_in_isoscalar, t_in_isovector,
                 a_omega, mass_omega, decay_rate_omega,
                 a_omega_prime, mass_omega_prime, decay_rate_omega_prime,
                 a_omega_double_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
                 a_phi, mass_phi, decay_rate_phi,
                 a_phi_prime, mass_phi_prime, decay_rate_phi_prime,
                 mass_phi_double_prime, decay_rate_phi_double_prime,
                 a_rho, mass_rho, decay_rate_rho,
                 a_rho_prime, mass_rho_prime, decay_rate_rho_prime,
                 a_rho_double_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
                 mass_rho_triple_prime, decay_rate_rho_triple_prime):
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self._data = [
            Parameter(name='t_in_isoscalar', value=t_in_isoscalar, is_fixed=False),
            Parameter(name='t_in_isovector', value=t_in_isovector, is_fixed=False),
            Parameter(name='a_omega', value=a_omega, is_fixed=False),
            Parameter(name='mass_omega', value=mass_omega, is_fixed=False),
            Parameter(name='decay_rate_omega', value=decay_rate_omega, is_fixed=False),
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
            Parameter(name='a_rho', value=a_rho, is_fixed=False),
            Parameter(name='mass_rho', value=mass_rho, is_fixed=False),
            Parameter(name='decay_rate_rho', value=decay_rate_rho, is_fixed=False),
            Parameter(name='a_rho_prime', value=a_rho_prime, is_fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, is_fixed=False),
            Parameter(name='a_rho_double_prime', value=a_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, is_fixed=False),
            Parameter(name='mass_rho_triple_prime', value=mass_rho_triple_prime, is_fixed=False),
            Parameter(name='decay_rate_rho_triple_prime', value=decay_rate_rho_triple_prime, is_fixed=False),
        ]

    def _find(self, name):
        for index, parameter in enumerate(self._data):
            if parameter.name == name:
                return index, parameter
        raise KeyError(f'No such key: {name}')

    def __getitem__(self, item):
        _, parameter = self._find(item)
        return parameter

    def __setitem__(self, key, new_value):
        index, parameter = self._find(key)
        if not isinstance(new_value, Parameter):
            raise TypeError('Bad type: new_value must be of type Parameter!')
        if not parameter.name == new_value.name:
            raise ValueError('When setting new parameters names must be preserved!')
        self._data[index] = new_value

    def set_value(self, parameter_name, new_parameter_value):
        index, old_par = self._find(parameter_name)
        new_par = Parameter(name=parameter_name, value=new_parameter_value, is_fixed=old_par.is_fixed)
        self._data[index] = new_par

    def fix_parameters(self, names):
        for name in names:
            index, parameter = self._find(name)
            self._data[index] = Parameter(parameter.name, parameter.value, True)

    def release_parameters(self, names):
        for name in names:
            index, parameter = self._find(name)
            self._data[index] = Parameter(parameter.name, parameter.value, False)

    def release_all_parameters(self):
        for index, parameter in enumerate(self._data):
            self._data[index] = Parameter(parameter.name, parameter.value, False)

    def get_fixed_values(self):
        return [parameter.value for parameter in self._data if parameter.is_fixed]

    def get_free_values(self):
        return [parameter.value for parameter in self._data if not parameter.is_fixed]

    def get_all_values(self):
        return list(map(lambda p: p.value, self._data))

    def update_free_values(self, new_values):
        free_parameters = [(i, parameter) for i, parameter
                           in enumerate(self._data) if not parameter.is_fixed]
        if len(free_parameters) != len(new_values):
            raise ValueError(f'Wrong number of new values: got {len(new_values)}, requires {len(free_parameters)}.')
        for (i, parameter), new_value in zip(free_parameters, new_values):
            self._data[i] = Parameter(parameter.name, new_value, parameter.is_fixed)

    def __iter__(self):
        for parameter in self._data:
            yield parameter

    def to_list(self):
        return list(self._data)

    def get_bounds_for_free_parameters(self):
        full_bounds = self.get_model_parameters_bounds()
        lower_bounds = []
        upper_bounds = []
        for parameter in self._data:
            if not parameter.is_fixed:
                bounds = full_bounds[parameter.name]
                lower_bounds.append(bounds['lower'])
                upper_bounds.append(bounds['upper'])
        return lower_bounds, upper_bounds

    def get_model_parameters_bounds(self):
        """
        There are no upper bounds. The branch points t_in_isoscalar and t_in_isovector
        must lie above the values of t_0_isoscalar and t_0_isovector, respectively.
        Decay rates must be non-negative and squared masses must lie above their respective t_0 treshold.

        """
        lower_mass_bound_isoscalar = math.sqrt(self.t_0_isoscalar)
        lower_mass_bound_isovector = math.sqrt(self.t_0_isovector)
        return {
            't_in_isoscalar': {'lower': self.t_0_isoscalar, 'upper': np.inf},
            't_in_isovector': {'lower': self.t_0_isovector, 'upper': np.inf},
            'a_omega': {'lower': -np.inf, 'upper': np.inf},
            'mass_omega': {'lower': lower_mass_bound_isoscalar, 'upper': np.inf},
            'decay_rate_omega': {'lower': 0.0, 'upper': np.inf},
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
            'a_rho': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho': {'lower': 0.0, 'upper': np.inf},
            'a_rho_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_prime': {'lower': 0.0, 'upper': np.inf},
            'a_rho_double_prime': {'lower': -np.inf, 'upper': np.inf},
            'mass_rho_double_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_double_prime': {'lower': 0.0, 'upper': np.inf},
            'mass_rho_triple_prime': {'lower': lower_mass_bound_isovector, 'upper': np.inf},
            'decay_rate_rho_triple_prime': {'lower': 0.0, 'upper': np.inf},
        }
