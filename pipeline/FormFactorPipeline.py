from typing import List, Type, Union

from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from pipeline.Pipeline import Pipeline
from task.KaonFormFactorTask import KaonFormFactorTask


class FormFactorPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[KaonFormFactorTask]],
                 t_values_charged: List[float], form_factors_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], form_factors_neutral: List[float], errors_neutral: List[float],
                 reports_dir: str, plot: bool = True, use_handpicked_bounds: bool = True) -> None:

        super().__init__(name, parameters, tasks,
                         t_values_charged, form_factors_charged, errors_charged,
                         t_values_neutral, form_factors_neutral, errors_neutral,
                         reports_dir, plot, use_handpicked_bounds)

    def _create_task(self, task_name: str, task_class: type(KaonFormFactorTask)) -> KaonFormFactorTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )
