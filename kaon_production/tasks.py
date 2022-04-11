import math
import random

from kaon_production.ModelParameters import ModelParameters
from kaon_production.Task import Task
from kaon_production.utils import make_partial_cross_section_for_parameters


class TaskFullFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


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
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstants(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega', 'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho', 'a_rho_prime', 'a_rho_double_prime',
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstantsAndMassesOfSelectedResonances(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega', 'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho', 'a_rho_prime', 'a_rho_double_prime', 'mass_omega', 'mass_phi', 'mass_rho'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixMassesOfSelectedResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['mass_omega', 'mass_phi', 'mass_rho'])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFitLowEnergies(Task):

    def __init__(self, name: str, parameters: ModelParameters, ts, css, errors,
                 t_0_isoscalar, t_0_isovector, reports_dir, plot=True, threshold=1.2):
        super().__init__(name, parameters, ts, css, errors, t_0_isoscalar, t_0_isovector, reports_dir, plot)
        assert threshold > 0
        self.threshold = threshold

    def _set_up(self):
        threshold_mass = math.sqrt(self.threshold)
        for resonance_name in ['omega', 'omega_prime', 'omega_double_prime',
                               'phi', 'phi_prime', 'phi_double_prime',
                               'rho', 'rho_prime', 'rho_double_prime', 'rho_triple_prime']:
            if self.parameters[f'mass_{resonance_name}'].value <= threshold_mass:
                self.parameters.release_parameters(self._parameter_tied_to_resonance(resonance_name))
            else:
                self.parameters.fix_parameters(self._parameter_tied_to_resonance(resonance_name))
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self._crop_data_for_fitting()

    def _crop_data_for_fitting(self):
        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *[(t, ff, err) for t, ff, err in zip(self.ts_fit, self.css_fit, self.errors_fit) if t <= self.threshold]
        )

    @staticmethod
    def _parameter_tied_to_resonance(resonance_name):
        par_names = [f'mass_{resonance_name}', f'decay_rate_{resonance_name}']
        if resonance_name not in ['phi_double_prime', 'rho_triple_prime']:
            par_names.append(f'a_{resonance_name}')
        return par_names


class TaskFitHighEnergies(Task):

    def __init__(self, name: str, parameters: ModelParameters, ts, css, errors,
                 t_0_isoscalar, t_0_isovector, reports_dir, plot=True, threshold=1.2):
        super().__init__(name, parameters, ts, css, errors, t_0_isoscalar, t_0_isovector, reports_dir, plot)
        assert threshold > 0
        self.threshold = threshold

    def _set_up(self):
        threshold_mass = math.sqrt(self.threshold)
        for resonance_name in ['omega', 'omega_prime', 'omega_double_prime',
                               'phi', 'phi_prime', 'phi_double_prime',
                               'rho', 'rho_prime', 'rho_double_prime', 'rho_triple_prime']:
            if self.parameters[f'mass_{resonance_name}'].value > threshold_mass:
                self.parameters.release_parameters(self._parameter_tied_to_resonance(resonance_name))
            else:
                self.parameters.fix_parameters(self._parameter_tied_to_resonance(resonance_name))
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self._crop_data_for_fitting()

    def _crop_data_for_fitting(self):
        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *[(t, ff, err) for t, ff, err in zip(self.ts_fit, self.css_fit, self.errors_fit) if t > self.threshold]
        )

    @staticmethod
    def _parameter_tied_to_resonance(resonance_name):
        par_names = [f'mass_{resonance_name}', f'decay_rate_{resonance_name}']
        if resonance_name not in ['phi_double_prime', 'rho_triple_prime']:
            par_names.append(f'a_{resonance_name}')
        return par_names


class TaskFitOnRandomSubsetOfData(Task):

    THRESHOLD = 0.8 # probability of retaining a datapoint

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )
        self._crop_data_for_fitting()

    def _crop_data_for_fitting(self):
        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *[(t, ff, err) for t, ff, err in zip(self.ts_fit, self.css_fit, self.errors_fit)
              if random.random() < self.THRESHOLD]
        )


class TaskOnlyThresholdsFit(Task):
    def _set_up(self):
        self.parameters.fix_all_parameters()
        self.parameters.release_parameters(['t_in_isoscalar', 't_in_isovector'])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedResonancesAndThresholdsFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'mass_omega', 'decay_rate_omega', 'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_omega_double_prime', 'decay_rate_omega_double_prime', 'mass_phi', 'decay_rate_phi',
            'mass_phi_prime', 'decay_rate_phi_prime', 'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'mass_rho', 'decay_rate_rho', 'mass_rho_prime', 'decay_rate_rho_prime',
            'mass_rho_double_prime', 'decay_rate_rho_double_prime', 'mass_rho_triple_prime',
            'decay_rate_rho_triple_prime', 't_in_isoscalar', 't_in_isovector',
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )