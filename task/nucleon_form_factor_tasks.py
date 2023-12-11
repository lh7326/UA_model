from task.NucleonFormFactorTask import NucleonFormFactorTask
from common.utils import make_partial_form_factor_for_parameters


class TaskFullFit(NucleonFormFactorTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)


class TaskFixAccordingToParametersFit(NucleonFormFactorTask):

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)
