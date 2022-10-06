from task.CrossSectionTask import CrossSectionTask
from common.utils import make_partial_cross_section_for_parameters


class TaskFullFit(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFullFitOnlyCharged(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixedResonancesFit(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_resonances()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedResonancesFitOnlyCharged(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_resonances()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )
        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixedCouplingConstants(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        coupling_constants = [p.name for p in self.parameters if p.name[:2] == 'a_']
        self.parameters.fix_parameters(coupling_constants)
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstantsOnlyCharged(CrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        coupling_constants = [p.name for p in self.parameters if p.name[:2] == 'a_']
        self.parameters.fix_parameters(coupling_constants)
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixAccordingToParametersFit(CrossSectionTask):

    def _set_up(self):
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixAccordingToParametersFitOnlyCharged(CrossSectionTask):

    def _set_up(self):
        self.partial_f = make_partial_cross_section_for_parameters(
            self.k_meson_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )
