from typing import List, Tuple, Type, Union

from kaon_production.data import KaonDatapoint
from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from pipeline.Pipeline import Pipeline
from task.KaonCrossSectionTask import KaonCrossSectionTask


class KaonCrossSectionPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[KaonCrossSectionTask]],
                 t_values_charged: List[float], cross_sections_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], cross_sections_neutral: List[float], errors_neutral: List[float],
                 k_meson_mass: float, alpha: float, hc_squared: float, reports_dir: str,
                 plot: bool = True, use_handpicked_bounds: bool = True) -> None:
        ts, css, errors = self._prepare_data(
            t_values_charged, cross_sections_charged, errors_charged,
            t_values_neutral, cross_sections_neutral, errors_neutral,
        )

        super().__init__(name, parameters, tasks,
                         ts, css, errors,
                         reports_dir, plot, use_handpicked_bounds)
        self.k_meson_mass = k_meson_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _create_task(self, task_name: str, task_class: type(KaonCrossSectionTask)) -> KaonCrossSectionTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.k_meson_mass, self.alpha, self.hc_squared,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

    @staticmethod
    def _prepare_data(
            ts_charged: List[float], css_charged: List[float], errors_charged: List[float],
            ts_neutral: List[float], css_neutral: List[float], errors_neutral: List[float],
    ) -> Tuple[List[KaonDatapoint], List[float], List[float]]:
        ts = [KaonDatapoint(t, True, True) for t in ts_charged]
        ys = list(css_charged)
        errors = list(errors_charged)
        ts += [KaonDatapoint(t, False, True) for t in ts_neutral]
        ys += list(css_neutral)
        errors += list(errors_neutral)

        ts, ys, errors = zip(
            *sorted(
                zip(ts, ys, errors),
                key=lambda tup: tup[0].t,
            )
        )
        return ts, ys, errors
