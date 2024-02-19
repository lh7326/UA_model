import random
from task.KaonCombinedTask import KaonCombinedTask
from common.utils import make_partial_ff_or_cs_for_parameters

# TODO: is it possible to merge tasks for kaons and nucleons?


class TaskFullFit(KaonCombinedTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixedCouplingConstants(KaonCombinedTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        coupling_constants = [p.name for p in self.parameters if p.name[:2] == 'a_']
        self.parameters.fix_parameters(coupling_constants)
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFixAccordingToParametersFit(KaonCombinedTask):

    def _set_up(self):
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )


class TaskFullFitOnlyTimelike(KaonCombinedTask):

    def _set_up(self):
        self.parameters.release_all_parameters()
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda data_tuple: data_tuple[0].t > 0.0,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixAccordingToParametersFitOnlyTimelike(KaonCombinedTask):

    def _set_up(self):
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda data_tuple: data_tuple[0].t > 0.0,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )


class TaskFixAccordingToParametersFitOnlyTimelikeSubsetOfDataset(KaonCombinedTask):

    def _set_up(self):
        self.partial_f = make_partial_ff_or_cs_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(
            *filter(lambda data_tuple: data_tuple[0].t > 0.0,
                    zip(self.ts_fit, self.ys_fit, self.errors_fit),
                    )
        )

        self.ts_fit, self.ys_fit, self.errors_fit = zip(*sorted(
            random.sample(list(zip(self.ts_fit, self.ys_fit, self.errors_fit)), k=(len(self.ts_fit) // 2)),
            key=lambda t: t[0].t
        ))
