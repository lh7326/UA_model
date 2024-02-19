from typing import List, Optional, Tuple, Union
from random import sample

from pipeline.KaonCombinedPipeline import KaonCombinedPipeline
from model_parameters import KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedSelected
from task.kaon_combined_tasks import (TaskFixAccordingToParametersFit, TaskFullFit,
                                      TaskFixAccordingToParametersFitOnlyTimelike, TaskFullFitOnlyTimelike,
                                      TaskFixAccordingToParametersFitOnlyTimelikeSubsetOfDataset)


class KaonCombinedIterativePipeline(KaonCombinedPipeline):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersB,
                                   KaonParametersSimplified, KaonParametersFixedSelected],
                 k_meson_mass: float, alpha: float, hc_squared: float, reports_dir: str,
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
                 plot: bool = True, use_handpicked_bounds: bool = True,
                 nr_free_params: Tuple[int, ...] = (3, 5, 7, 10),
                 nr_iterations: Tuple[int, ...] = (10, 20, 20, 10),
                 nr_initial_rounds_with_fixed_resonances: int = 0,
                 nr_initial_rounds_on_partial_dataset: int = 0,
                 fit_on_timelike_data_only: bool = False) -> None:

        super().__init__(name, parameters, [], k_meson_mass, alpha, hc_squared, reports_dir,
                         t_cs_values_charged, cross_sections_charged, cs_errors_charged,
                         t_cs_values_neutral, cross_sections_neutral, cs_errors_neutral,
                         t_ff_values_charged, form_factors_charged, ff_errors_charged,
                         t_ff_values_neutral, form_factors_neutral, ff_errors_neutral,
                         plot, use_handpicked_bounds)

        self.free_params_numbers = []
        for free_pars, repetitions in zip(nr_free_params, nr_iterations):
            self.free_params_numbers.extend([free_pars] * repetitions)
        self.nr_initial_rounds_with_fixed_resonances = nr_initial_rounds_with_fixed_resonances
        self.nr_initial_rounds_with_partial_dataset = nr_initial_rounds_on_partial_dataset
        self.fit_on_timelike_data_only = fit_on_timelike_data_only

    def run(self) -> dict:
        self._log(f'Starting. Initial parameters: {self.parameters.to_list()}')
        for i, fp_num in enumerate(self.free_params_numbers):
            fix_resonances = (i < self.nr_initial_rounds_with_fixed_resonances)
            partial_dataset = (i < self.nr_initial_rounds_with_partial_dataset)
            free_params = self._randomly_freeze_parameters(fp_num, fix_resonances)
            self._log(f'Initializing Task#{i}. Free parameters: {free_params}')

            if self.fit_on_timelike_data_only and partial_dataset:
                task_class = TaskFixAccordingToParametersFitOnlyTimelikeSubsetOfDataset
            elif self.fit_on_timelike_data_only and not partial_dataset:
                task_class = TaskFixAccordingToParametersFitOnlyTimelike
            elif not self.fit_on_timelike_data_only and partial_dataset:
                raise NotImplementedError
            elif not self.fit_on_timelike_data_only and not partial_dataset:
                task_class = TaskFixAccordingToParametersFit
            else:
                raise 'Incorrect case syntax'

            task_name = f'Task#{i}:{task_class.__name__}'
            task = task_class(  # type: ignore
                task_name, self.parameters,
                self.ts, self.ys, self.errors,
                self.k_meson_mass, self.alpha, self.hc_squared,
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
        task_class = (TaskFullFitOnlyTimelike if self.fit_on_timelike_data_only
                      else TaskFullFit)
        task_name = f'Task#{len(self.free_params_numbers)}:{task_class.__name__}'
        task = task_class(  # type: ignore
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.k_meson_mass, self.alpha, self.hc_squared,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

        self._log(f'Running {task_name}')
        task.run()
        self._log(f'{task_name} report: {task.report}')
        self._update_best_fit(task)
        self.parameters = task.parameters

        self._log(f'Best fit: {self._best_fit}')
        final_chi_squared = task.report['chi_squared']
        final_chi_squared_on_training_set = task.report['chi_squared_on_training_set']
        self._log(f'Final fit: chi_squared={final_chi_squared};'
                  f'chi_squared_on_training_set={final_chi_squared_on_training_set};'
                  f' parameters={self.parameters.to_list()}')
        self._flush_report()
        self._save_report(str(len(self.free_params_numbers)), task.report)
        self._serialize_parameters(name='final_fit_parameters')
        return self._best_fit

    def _randomly_freeze_parameters(self, number_of_free_parameters, fix_resonances):
        self.parameters.release_all_parameters()  # this allows us to identify which parameters cannot be released
        if fix_resonances:
            self.parameters.fix_resonances()  # type: ignore
        names = [par.name for par in self.parameters if not par.is_fixed]
        chosen_names = sample(names, k=number_of_free_parameters)
        self.parameters.fix_all_parameters()
        self.parameters.release_parameters(chosen_names)
        return chosen_names
