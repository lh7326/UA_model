from typing import List, Tuple, Union
from random import sample

from kaon_production.Pipeline import Pipeline
from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from kaon_production.tasks import TaskFixAccordingToParametersFit, TaskFullFitOnlyCharged


class IterativePipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 t_values_charged: List[float], cross_sections_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], cross_sections_neutral: List[float], errors_neutral: List[float],
                 k_meson_mass: float, alpha: float, hc_squared: float,
                 t_0_isoscalar: float, t_0_isovector: float, reports_dir: str,
                 plot: bool = True, use_handpicked_bounds: bool = True,
                 nr_free_params: Tuple[int, ...] = (3, 5, 7, 10),
                 nr_iterations: Tuple[int, ...] = (10, 20, 20, 10),
                 nr_initial_rounds_with_fixed_resonances: int = 0) -> None:

        super().__init__(name, parameters, [], t_values_charged, cross_sections_charged, errors_charged,
                         t_values_neutral, cross_sections_neutral, errors_neutral, k_meson_mass, alpha, hc_squared,
                         t_0_isoscalar, t_0_isovector, reports_dir, plot, use_handpicked_bounds)

        self.free_params_numbers = []
        for free_pars, repetitions in zip(nr_free_params, nr_iterations):
            self.free_params_numbers.extend([free_pars] * repetitions)
        self.nr_initial_rounds_with_fixed_resonances = nr_initial_rounds_with_fixed_resonances

    def run(self) -> dict:
        self._log(f'Starting. Initial parameters: {self.parameters.to_list()}')
        for i, fp_num in enumerate(self.free_params_numbers):
            fix_resonances = (i < self.nr_initial_rounds_with_fixed_resonances)
            free_params = self._randomly_freeze_parameters(fp_num, fix_resonances)
            self._log(f'Initializing Task#{i}. Free parameters: {free_params}')
            task_name = f'Task#{i}:{TaskFixAccordingToParametersFit.__name__}'
            task = TaskFixAccordingToParametersFit(
                task_name, self.parameters,
                self.ts, self.cross_sections, self.errors,
                self.k_meson_mass, self.alpha, self.hc_squared,
                self.t_0_isoscalar, self.t_0_isovector,
                self.reports_dir, self.plot, self.use_handpicked_bounds
            )

            self._log(f'Running {task_name}')
            task.run()
            self._log(f'{task_name} report: {task.report}')
            self._update_best_fit(task)

            self.parameters = task.parameters
            self._flush_report()
            self._save_report(str(i), task.report)

        self._log(f'Initializing Task#{len(self.free_params_numbers)}. Full fit.')
        task_name = f'Task#{len(self.free_params_numbers)}:{TaskFullFitOnlyCharged.__name__}'
        task = TaskFullFitOnlyCharged(
            task_name, self.parameters,
            self.ts, self.cross_sections, self.errors,
            self.k_meson_mass, self.alpha, self.hc_squared,
            self.t_0_isoscalar, self.t_0_isovector,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

        self._log(f'Running {task_name}')
        task.run()
        self._log(f'{task_name} report: {task.report}')
        self._update_best_fit(task)
        self.parameters = task.parameters

        self._log(f'Best fit: {self._best_fit}')
        self._flush_report()
        self._save_report(str(len(self.free_params_numbers)), task.report)
        return self._best_fit

    def _randomly_freeze_parameters(self, number_of_free_parameters, fix_resonances):
        self.parameters.release_all_parameters()  # this allows us to identify which parameters cannot be released
        if fix_resonances:
            self.parameters.fix_resonances()
        names = [par.name for par in self.parameters if not par.is_fixed]
        chosen_names = sample(names, k=number_of_free_parameters)
        self.parameters.fix_all_parameters()
        self.parameters.release_parameters(chosen_names)
        return chosen_names
