from task.KaonCrossSectionTask import KaonCrossSectionTask
from common.utils import make_partial_cross_section_for_parameters

# TODO: is it possible to merge tasks for kaons and nucleons?


class TaskFullFit(KaonCrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_cross_section_for_parameters(
            self.alpha, self.hc_squared, self.parameters,
            charged_kaon_mass=self.charged_kaon_mass,
            neutral_kaon_mass=self.neutral_kaon_mass,
        )


class TaskFixedCouplingConstants(KaonCrossSectionTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        coupling_constants = [p.name for p in self.parameters if p.name[:2] == 'a_']
        self.parameters.fix_parameters(coupling_constants)
        self.partial_f = make_partial_cross_section_for_parameters(
            self.alpha, self.hc_squared, self.parameters,
            charged_kaon_mass=self.charged_kaon_mass,
            neutral_kaon_mass=self.neutral_kaon_mass,
        )


class TaskFixAccordingToParametersFit(KaonCrossSectionTask):

    def _set_up(self):
        self.partial_f = make_partial_cross_section_for_parameters(
            self.alpha, self.hc_squared, self.parameters,
            charged_kaon_mass=self.charged_kaon_mass,
            neutral_kaon_mass=self.neutral_kaon_mass,
        )
