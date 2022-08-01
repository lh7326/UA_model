from abc import ABC, abstractmethod
from typing import List, Type, Union
import os.path

from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from task.Task import Task
from kaon_production.data import Datapoint


class Pipeline(ABC):

    def __init__(self, name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 tasks: List[Type[Task]],
                 t_values_charged: List[float], ys_charged: List[float], errors_charged: List[float],
                 t_values_neutral: List[float], ys_neutral: List[float], errors_neutral: List[float],
                 reports_dir: str, plot: bool = True, use_handpicked_bounds: bool = True) -> None:
        self.name = name
        self.parameters = parameters
        self.tasks = tasks
        self.t_values_charged = t_values_charged
        self.ys_charged = ys_charged
        self.errors_charged = errors_charged
        self.t_values_neutral = t_values_neutral
        self.ys_neutral = ys_neutral
        self.errors_neutral = errors_neutral
        self.reports_dir = os.path.join(reports_dir, name)
        self.plot = plot
        self.use_handpicked_bounds = use_handpicked_bounds

        self.ts = None
        self.errors = None
        self.ys = None
        self._prepare_data()

        self._report = f'Report {name}:\n'
        self._set_up_reports_directory()
        self._best_fit = {'chi_squared': None, 'name': None, 'parameters': None, 'parameters_list': None}

    def run(self) -> dict:
        self._log(f'Starting. Initial parameters: {self.parameters.to_list()}')
        for i, task_class in enumerate(self.tasks):
            self._log(f'Initializing Task#{i}. Parameters: {self.parameters.to_list()}')
            task_name = f'Task#{i}:{task_class.__name__}'
            task = self._create_task(task_name, task_class)

            self._log(f'Running {task_name}')
            task.run()
            self._log(f'{task_name} report: {task.report}')
            self._update_best_fit(task)

            self.parameters = task.parameters
            self.parameters.release_all_parameters()
            self._flush_report()
            self._save_report(str(i), task.report)
        self._log(f'Best fit: {self._best_fit}')
        self._flush_report()
        return self._best_fit

    def _log(self, msg: str) -> None:
        print(f'Pipeline {self.name}: {msg}\n')
        self._report += (msg + '\n')

    def _set_up_reports_directory(self) -> None:
        if os.path.exists(self.reports_dir):
            raise ValueError(f'Directory already exists: {self.reports_dir}')
        os.mkdir(self.reports_dir)

    def _flush_report(self) -> None:
        filepath = os.path.join(self.reports_dir, 'report.txt')
        with open(filepath, 'a') as f:
            f.write(self._report)
        self._report = '\n'

    def _save_report(self, name: str, report: dict) -> None:
        filename = f'report_{name}.txt'
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, 'w') as f:
            f.write(str(report))

    def _update_best_fit(self, task: Task) -> None:
        if not task.report['chi_squared']:
            return None
        current = task.report['chi_squared']
        best_so_far = self._best_fit.get('chi_squared', None)
        if not best_so_far or current < best_so_far:
            self._best_fit = {
                'chi_squared': current,
                'name': f'{self.name}:{task.name}',
                'parameters': task.parameters.to_list(),
                'parameters_list': task.parameters.get_ordered_values(),
                'covariance_matrix': task.report.get('covariance_matrix'),
                'parameter_errors': task.report.get('parameter_errors'),
            }

    def _prepare_data(self):
        self.ts = [Datapoint(t, True) for t in self.t_values_charged]
        self.ys = list(self.ys_charged)
        self.errors = list(self.errors_charged)
        self.ts += [Datapoint(t, False) for t in self.t_values_neutral]
        self.ys += list(self.ys_neutral)
        self.errors += list(self.errors_neutral)

        self.ts, self.ys, self.errors = zip(
            *sorted(
                zip(self.ts, self.ys, self.errors),
                key=lambda tup: tup[0].t,
            )
        )

    @abstractmethod
    def _create_task(self, task_name: str, task_class: type(Task)) -> Task:
        pass
