from task.KaonFormFactorTask import KaonFormFactorTask
from kaon_production.utils import make_partial_form_factor_for_parameters


class TaskFullFit(KaonFormFactorTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)


class TaskFullFitOnlyCharged(KaonFormFactorTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixAccordingToParametersFit(KaonFormFactorTask):

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)


class TaskFixAccordingToParametersFitOnlyCharged(KaonFormFactorTask):

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )
