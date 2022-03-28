from collections import namedtuple
import math
import numpy as np
from scipy.optimize import curve_fit

from plotting.plot_fit import plot_ff_fit
from kaon_production.function import function_form_factor
from kaon_production.utils import report_fit

Parameter = namedtuple('Parameter', 'name value fixed')


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
            Parameter(name='t_in_isoscalar', value=t_in_isoscalar, fixed=False),
            Parameter(name='t_in_isovector', value=t_in_isovector, fixed=False),
            Parameter(name='a_omega', value=a_omega, fixed=False),
            Parameter(name='mass_omega', value=mass_omega, fixed=False),
            Parameter(name='decay_rate_omega', value=decay_rate_omega, fixed=False),
            Parameter(name='a_omega_prime', value=a_omega_prime, fixed=False),
            Parameter(name='mass_omega_prime', value=mass_omega_prime, fixed=False),
            Parameter(name='decay_rate_omega_prime', value=decay_rate_omega_prime, fixed=False),
            Parameter(name='a_omega_double_prime', value=a_omega_double_prime, fixed=False),
            Parameter(name='mass_omega_double_prime', value=mass_omega_double_prime, fixed=False),
            Parameter(name='decay_rate_omega_double_prime', value=decay_rate_omega_double_prime, fixed=False),
            Parameter(name='a_phi', value=a_phi, fixed=False),
            Parameter(name='mass_phi', value=mass_phi, fixed=False),
            Parameter(name='decay_rate_phi', value=decay_rate_phi, fixed=False),
            Parameter(name='a_phi_prime', value=a_phi_prime, fixed=False),
            Parameter(name='mass_phi_prime', value=mass_phi_prime, fixed=False),
            Parameter(name='decay_rate_phi_prime', value=decay_rate_phi_prime, fixed=False),
            Parameter(name='mass_phi_double_prime', value=mass_phi_double_prime, fixed=False),
            Parameter(name='decay_rate_phi_double_prime', value=decay_rate_phi_double_prime, fixed=False),
            Parameter(name='a_rho', value=a_rho, fixed=False),
            Parameter(name='mass_rho', value=mass_rho, fixed=False),
            Parameter(name='decay_rate_rho', value=decay_rate_rho, fixed=False),
            Parameter(name='a_rho_prime', value=a_rho_prime, fixed=False),
            Parameter(name='mass_rho_prime', value=mass_rho_prime, fixed=False),
            Parameter(name='decay_rate_rho_prime', value=decay_rate_rho_prime, fixed=False),
            Parameter(name='a_rho_double_prime', value=a_rho_double_prime, fixed=False),
            Parameter(name='mass_rho_double_prime', value=mass_rho_double_prime, fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=decay_rate_rho_double_prime, fixed=False),
            Parameter(name='mass_rho_triple_prime', value=mass_rho_triple_prime, fixed=False),
            Parameter(name='decay_rate_rho_triple_prime', value=decay_rate_rho_triple_prime, fixed=False),
        ]

    def _find(self, name):
        for index, parameter in enumerate(self._data):
            if parameter.name == name:
                return index, parameter
        raise KeyError

    def __getitem__(self, item):
        _, parameter = self._find(item)
        return parameter

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
        return [parameter.value for parameter in self._data if parameter.fixed]

    def get_free_values(self):
        return [parameter.value for parameter in self._data if not parameter.fixed]

    def get_all_values(self):
        return list(map(lambda p: p.value, self._data))

    def update_free_values(self, new_values):
        free_parameters = [(i, parameter) for i, parameter
                           in enumerate(self._data) if not parameter.fixed]
        assert len(free_parameters) == new_values
        for (i, parameter), new_value in zip(free_parameters, new_values):
            self._data[i] = Parameter(parameter.name, new_value, parameter.fixed)

    def __iter__(self):
        for parameter in self._data:
            yield parameter
        raise StopIteration

    def to_list(self):
        return list(self._data)

    def get_bounds_for_free_parameters(self):
        full_bounds = self.get_model_parameters_bounds()
        lower_bounds = []
        upper_bounds = []
        for parameter in self._data:
            if not parameter.fixed:
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


class Task:

    def __init__(self, name: str, parameters: ModelParameters, ts, ffs, errors,
                 t_0_isoscalar, t_0_isovector, plot=True):
        self.name = name
        self.parameters = parameters
        self.partial_f = None  # prepared in the _setup method
        self.ts = ts
        self.ffs = ffs
        self.errors = errors
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self.should_plot = plot

    def run(self):
        print(f'Running task {self.name}...')
        self._set_up()
        opt_params = self._fit()
        self.parameters.update_free_values(opt_params)
        report_fit(self.ts, self.ffs, self.errors, opt_params, self.partial_f, f'Task {self.name}')
        if self.should_plot:
            self._plot(opt_params)
        return self.parameters

    def _fit(self):
        opt_params, _ = curve_fit(
            f=self.partial_f,
            xdata=self.ts,
            ydata=self.ffs,
            p0=self.parameters.get_free_values(),
            sigma=self.errors,
            absolute_sigma=False,
            bounds=self.parameters.get_bounds_for_free_parameters(),
            maxfev=5000,
        )
        return opt_params

    def _plot(self, opt_params):
        plot_ff_fit(self.ts, self.ffs, self.errors, self.partial_f,
                    opt_params, f'Task {self.name}')

    def _set_up(self):
        raise NotImplementedError


class TaskFullFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_for_parameters(self.parameters)


class TaskFixedResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'mass_omega', 'decay_rate_omega', 'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_omega_double_prime', 'decay_rate_omega_double_prime', 'mass_phi', 'decay_rate_phi',
            'mass_phi_prime', 'decay_rate_phi_prime', 'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'mass_rho', 'decay_rate_rho', 'mass_rho_prime', 'decay_rate_rho_prime',
            'mass_rho_double_prime', 'decay_rate_rho_double_prime', 'mass_rho_triple_prime',
            'decay_rate_rho_triple_prime'
        ])
        self.partial_f = make_partial_for_parameters(self.parameters)


class TaskFixedCouplingConstants(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega', 'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho', 'a_rho_prime', 'a_rho_double_prime',
        ])
        self.partial_f = make_partial_for_parameters(self.parameters)


class TaskFitLowEnergies(Task):
    pass

class TaskFitHighEnergies(Task):
    pass

class RandomlyPerturbParameters(Task):
    pass

def make_partial_for_parameters(parameters: ModelParameters):
    def _from_parameters_or_arguments(name, args):
        parameter = parameters[name]
        if parameter.fixed:
            return parameter.value
        else:
            return args.pop()

    def partial_f(ts, *args):

        # WARNING: the order of the commands below is important!
        decay_rate_rho_triple_prime = _from_parameters_or_arguments('decay_rate_rho_triple_prime', args)
        mass_rho_triple_prime = _from_parameters_or_arguments('mass_rho_triple_prime', args)
        decay_rate_rho_double_prime = _from_parameters_or_arguments('decay_rate_rho_double_prime', args)
        mass_rho_double_prime = _from_parameters_or_arguments('mass_rho_double_prime', args)
        a_rho_double_prime = _from_parameters_or_arguments('a_rho_double_prime', args)
        decay_rate_rho_prime = _from_parameters_or_arguments('decay_rate_rho_prime', args)
        mass_rho_prime = _from_parameters_or_arguments('mass_rho_prime', args)
        a_rho_prime = _from_parameters_or_arguments('a_rho_prime', args)
        decay_rate_rho = _from_parameters_or_arguments('decay_rate_rho', args)
        mass_rho = _from_parameters_or_arguments('mass_rho', args)
        a_rho = _from_parameters_or_arguments('a_rho', args)
        decay_rate_phi_double_prime = _from_parameters_or_arguments('decay_rate_phi_double_prime', args)
        mass_phi_double_prime = _from_parameters_or_arguments('mass_phi_double_prime', args)
        decay_rate_phi_prime = _from_parameters_or_arguments('decay_rate_phi_prime', args)
        mass_phi_prime = _from_parameters_or_arguments('mass_phi_prime', args)
        a_phi_prime = _from_parameters_or_arguments('a_phi_prime', args)
        decay_rate_phi = _from_parameters_or_arguments('decay_rate_phi', args)
        mass_phi = _from_parameters_or_arguments('mass_phi', args)
        a_phi = _from_parameters_or_arguments('a_phi', args)
        decay_rate_omega_double_prime = _from_parameters_or_arguments('decay_rate_omega_double_prime', args)
        mass_omega_double_prime = _from_parameters_or_arguments('mass_omega_double_prime', args)
        a_omega_double_prime = _from_parameters_or_arguments('a_omega_double_prime', args)
        decay_rate_omega_prime = _from_parameters_or_arguments('decay_rate_omega_prime', args)
        mass_omega_prime = _from_parameters_or_arguments('mass_omega_prime', args)
        a_omega_prime = _from_parameters_or_arguments('a_omega_prime', args)
        decay_rate_omega = _from_parameters_or_arguments('decay_omega_phi', args)
        mass_omega = _from_parameters_or_arguments('mass_omega', args)
        a_omega = _from_parameters_or_arguments('a_omega', args)
        t_in_isovector = _from_parameters_or_arguments('t_in_isovector', args)
        t_in_isoscalar = _from_parameters_or_arguments('t_in_isoscalar', args)

        assert not args

        return function_form_factor(
            ts, parameters.t_0_isoscalar, parameters.t_0_isovector,
            t_in_isoscalar, t_in_isovector, a_omega, a_omega_prime, a_omega_double_prime,
            a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime, mass_omega, decay_rate_omega,
            mass_omega_prime, decay_rate_omega_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
            mass_phi, decay_rate_phi, mass_phi_prime, decay_rate_phi_prime, mass_phi_double_prime,
            decay_rate_phi_double_prime, mass_rho, decay_rate_rho, mass_rho_prime,
            decay_rate_rho_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
            mass_rho_triple_prime, decay_rate_rho_triple_prime)

    return partial_f
