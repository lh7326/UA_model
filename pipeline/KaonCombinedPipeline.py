from typing import List, Optional, Tuple, Type, Union

from kaon_production.data import KaonDatapoint
from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from pipeline.Pipeline import Pipeline
from task.KaonCombinedTask import KaonCombinedTask


class KaonCombinedPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[KaonCombinedTask]],
                 charged_kaon_mass: float, neutral_kaon_mass: float,
                 alpha: float, hc_squared: float, reports_dir: str,
                 t_cs_values_charged: Optional[List[float]] = None,
                 cross_sections_charged: Optional[List[float]] = None,
                 cs_errors_charged: Optional[List[float]] = None,
                 t_cs_values_neutral: Optional[List[float]] = None,
                 cross_sections_neutral: Optional[List[float]] = None,
                 cs_errors_neutral: Optional[List[float]] = None,
                 t_ff_values_charged: Optional[List[float]] = None,
                 form_factors_charged: Optional[List[float]] = None,
                 ff_errors_charged: Optional[List[float]] = None,
                 t_ff_values_neutral: Optional[List[float]] = None,
                 form_factors_neutral: Optional[List[float]] = None,
                 ff_errors_neutral: Optional[List[float]] = None,
                 plot: bool = True, use_handpicked_bounds: bool = True) -> None:
        ts, ys, errors = self._prepare_data(
            t_cs_values_charged, cross_sections_charged, cs_errors_charged,
            t_cs_values_neutral, cross_sections_neutral, cs_errors_neutral,
            t_ff_values_charged, form_factors_charged, ff_errors_charged,
            t_ff_values_neutral, form_factors_neutral, ff_errors_neutral,
        )

        super().__init__(name, parameters, tasks,
                         ts, ys, errors,
                         reports_dir, plot, use_handpicked_bounds)
        self.charged_kaon_mass = charged_kaon_mass
        self.neutral_kaon_mass = neutral_kaon_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _create_task(self, task_name: str, task_class: type(KaonCombinedTask)) -> KaonCombinedTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.charged_kaon_mass, self.neutral_kaon_mass, self.alpha, self.hc_squared,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

    @staticmethod
    def _prepare_data(
            ts_cs_charged: Optional[List[float]], css_charged: Optional[List[float]],
            cs_errors_charged: Optional[List[float]],
            ts_cs_neutral: Optional[List[float]], css_neutral: Optional[List[float]],
            cs_errors_neutral: Optional[List[float]],
            ts_ff_charged: Optional[List[float]], ffs_charged: Optional[List[float]],
            ff_errors_charged: Optional[List[float]],
            ts_ff_neutral: Optional[List[float]], ffs_neutral: Optional[List[float]],
            ff_errors_neutral: Optional[List[float]],
    ) -> Tuple[List[KaonDatapoint], List[float], List[float]]:
        ts, ys, errors = [], [], []
        if ts_cs_charged is not None:
            assert len(ts_cs_charged) == len(css_charged or []) == len(cs_errors_charged or [])
            ts += [KaonDatapoint(t, True, True) for t in ts_cs_charged]
            ys += list(css_charged or [])
            errors += list(cs_errors_charged or [])
        if ts_cs_neutral is not None:
            assert len(ts_cs_neutral) == len(css_neutral or []) == len(cs_errors_neutral or [])
            ts += [KaonDatapoint(t, False, True) for t in ts_cs_neutral]
            ys += list(css_neutral or [])
            errors += list(cs_errors_neutral or [])
        if ts_ff_charged is not None:
            assert len(ts_ff_charged) == len(ffs_charged or []) == len(ff_errors_charged or [])
            ts += [KaonDatapoint(t, True, False) for t in ts_ff_charged]
            ys += list(ffs_charged or [])
            errors += list(ff_errors_charged or [])
        if ts_ff_neutral is not None:
            assert len(ts_ff_neutral) == len(ffs_neutral or []) == len(ff_errors_neutral or [])
            ts += [KaonDatapoint(t, False, False) for t in ts_ff_neutral]
            ys += list(ffs_neutral or [])
            errors += list(ff_errors_neutral or [])

        ts, ys, errors = zip(
            *sorted(
                zip(ts, ys, errors),
                key=lambda tup: tup[0].t,
            )
        )
        return ts, ys, errors
