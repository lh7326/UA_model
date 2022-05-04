from kaon_production.Task import Task
from kaon_production.utils import make_partial_cross_section_for_parameters


class TaskFullFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFullFitOnlyCharged(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.css_fit, self.errors_fit),
                    )
        )


class TaskFixedResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_omega_double_prime', 'decay_rate_omega_double_prime', 'mass_phi', 'decay_rate_phi',
            'mass_phi_prime', 'decay_rate_phi_prime', 'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'mass_rho_prime', 'decay_rate_rho_prime',
            'mass_rho_double_prime', 'decay_rate_rho_double_prime', 'mass_rho_triple_prime',
            'decay_rate_rho_triple_prime'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedResonancesFitOnlyCharged(Task):

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
        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.css_fit, self.errors_fit),
                    )
        )


class TaskFixedNearlyAllResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_phi', 'decay_rate_phi', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_rho_prime', 'decay_rate_rho_prime'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedNearlyAllResonancesFitOnlyCharged(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_phi', 'decay_rate_phi', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_rho_prime', 'decay_rate_rho_prime'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )
        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.css_fit, self.errors_fit),
                    )
        )


class TaskFixedCouplingConstantsAndNearlyAllResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho_prime', 'a_rho_double_prime',
            'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_phi', 'decay_rate_phi', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_rho_prime', 'decay_rate_rho_prime'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstants(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho_prime', 'a_rho_double_prime',
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstantsOnlyCharged(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega', 'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho', 'a_rho_prime', 'a_rho_double_prime',
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.css_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.css_fit, self.errors_fit),
                    )
        )


class TaskFixedCouplingConstantsAndMassesOfSelectedResonances(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_omega_prime', 'a_omega_double_prime', 'a_phi', 'a_phi_prime',
            'a_rho_prime', 'a_rho_double_prime', 'mass_phi'
        ])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixMassesOfSelectedResonancesFit(Task):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['mass_phi'])
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

class TaskFixAccordingToParametersFit(Task):

    def _set_up(self):
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )
