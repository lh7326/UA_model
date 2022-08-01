from task.FormFactorTask import FormFactorTask
from kaon_production.utils import make_partial_form_factor_for_parameters


class TaskFullFit(FormFactorTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)


class TaskFullFitOnlyCharged(FormFactorTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixAccordingToParametersFit(FormFactorTask):

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)


class TaskFixAccordingToParametersFitOnlyCharged(FormFactorTask):

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda t: t[0].is_charged,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )
