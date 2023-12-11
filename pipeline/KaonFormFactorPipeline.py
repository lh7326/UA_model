from typing import List, Tuple, Type, Union

from model_parameters import KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedSelected
from kaon_production.data import KaonDatapoint
from pipeline.Pipeline import Pipeline
from task.KaonFormFactorTask import KaonFormFactorTask


class KaonFormFactorPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersB,
                                   KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[KaonFormFactorTask]],
                 t_values_charged: List[float], form_factors_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], form_factors_neutral: List[float], errors_neutral: List[float],
                 reports_dir: str, plot: bool = True, use_handpicked_bounds: bool = True) -> None:

        ts, ys, errors = self._prepare_data(
            t_values_charged, form_factors_charged, errors_charged,
            t_values_neutral, form_factors_neutral, errors_neutral,
        )
        super().__init__(name, parameters, tasks,
                         ts, ys, errors,
                         reports_dir, plot, use_handpicked_bounds)

    def _create_task(self, task_name: str, task_class: type(KaonFormFactorTask)) -> KaonFormFactorTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

    @staticmethod
    def _prepare_data(
            ts_charged: List[float], ffs_charged: List[float], errors_charged: List[float],
            ts_neutral: List[float], ffs_neutral: List[float], errors_neutral: List[float],
    ) -> Tuple[List[KaonDatapoint], List[float], List[float]]:
        ts = [KaonDatapoint(t, True) for t in ts_charged]
        ys = list(ffs_charged)
        errors = list(errors_charged)
        ts += [KaonDatapoint(t, False) for t in ts_neutral]
        ys += list(ffs_neutral)
        errors += list(errors_neutral)

        ts, ys, errors = zip(
            *sorted(
                zip(ts, ys, errors),
                key=lambda tup: tup[0].t,
            )
        )
        return ts, ys, errors
