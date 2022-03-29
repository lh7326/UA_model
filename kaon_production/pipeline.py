import math
from scipy.optimize import curve_fit

from plotting.plot_fit import plot_ff_fit
from kaon_production.function import function_form_factor
from kaon_production.ModelParameters import ModelParameters
from kaon_production.utils import report_fit


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
        if parameter.is_fixed:
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
