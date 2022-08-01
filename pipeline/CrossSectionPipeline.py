from typing import List, Type, Union

from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from pipeline.Pipeline import Pipeline
from task.CrossSectionTask import CrossSectionTask


class CrossSectionPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[CrossSectionTask]],
                 t_values_charged: List[float], cross_sections_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], cross_sections_neutral: List[float], errors_neutral: List[float],
                 k_meson_mass: float, alpha: float, hc_squared: float, reports_dir: str,
                 plot: bool = True, use_handpicked_bounds: bool = True) -> None:

        super().__init__(name, parameters, tasks,
                         t_values_charged, cross_sections_charged, errors_charged,
                         t_values_neutral, cross_sections_neutral, errors_neutral,
                         reports_dir, plot, use_handpicked_bounds)
        self.k_meson_mass = k_meson_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _create_task(self, task_name: str, task_class: type(CrossSectionTask)) -> CrossSectionTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.k_meson_mass, self.alpha, self.hc_squared,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )
