import math

from kaon_production.ModelParameters import ModelParameters
from kaon_production.Task import Task
from kaon_production.utils import make_partial_for_parameters


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

    def __init__(self, name: str, parameters: ModelParameters, ts, ffs, errors,
                 t_0_isoscalar, t_0_isovector, threshold=1.2, plot=True):
        super().__init__(name, parameters, ts, ffs, errors, t_0_isoscalar, t_0_isovector, plot)
        assert threshold > 0
        self.threshold = threshold

    def _set_up(self):
        threshold_mass = math.sqrt(self.threshold)
        for resonance_name in ['omega', 'omega_prime', 'omega_double_prime',
                               'phi', 'phi_prime', 'phi_double_prime',
                               'rho', 'rho_prime', 'rho_double_prime', 'rho_triple_prime']:
            if self.parameters[f'mass_{resonance_name}'].value <= threshold_mass:
                self.parameters.release_parameters([resonance_name])
            else:
                self.parameters.fix_parameters([resonance_name])
        self.partial_f = make_partial_for_parameters(self.parameters)

        self.ts, self.ffs, self.errors = zip(
            *[(t, ff, err) for t, ff, err in zip(self.ts, self.ffs, self.errors) if t <= self.threshold]
        )


class TaskFitHighEnergies(Task):

    def __init__(self, name: str, parameters: ModelParameters, ts, ffs, errors,
                 t_0_isoscalar, t_0_isovector, threshold=1.2, plot=True):
        super().__init__(name, parameters, ts, ffs, errors, t_0_isoscalar, t_0_isovector, plot)
        assert threshold > 0
        self.threshold = threshold

    def _set_up(self):
        threshold_mass = math.sqrt(self.threshold)
        for resonance_name in ['omega', 'omega_prime', 'omega_double_prime',
                               'phi', 'phi_prime', 'phi_double_prime',
                               'rho', 'rho_prime', 'rho_double_prime', 'rho_triple_prime']:
            if self.parameters[f'mass_{resonance_name}'].value > threshold_mass:
                self.parameters.release_parameters([resonance_name])
            else:
                self.parameters.fix_parameters([resonance_name])
        self.partial_f = make_partial_for_parameters(self.parameters)

        self.ts, self.ffs, self.errors = zip(
            *[(t, ff, err) for t, ff, err in zip(self.ts, self.ffs, self.errors) if t > self.threshold]
        )
