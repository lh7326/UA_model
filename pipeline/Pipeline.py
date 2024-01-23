from abc import ABC, abstractmethod
from typing import List, Type, Union, Optional
import os.path

from model_parameters import ModelParameters
from task.Task import Task
from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint


class Pipeline(ABC):

    def __init__(self, name: str,
                 parameters: ModelParameters,
                 tasks: List[Type[Task]],
                 ts: Union[List[KaonDatapoint], List[NucleonDatapoint]],
                 ys: List[float], errors: List[float],
                 reports_dir: str, plot: bool = True, use_handpicked_bounds: bool = True) -> None:
        self.name = name
        self.parameters = parameters
        self.tasks = tasks
        self.ts = ts
        self.ys = ys
        self.errors = errors
        self.reports_dir = os.path.join(reports_dir, name)
        self.plot = plot
        self.use_handpicked_bounds = use_handpicked_bounds

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

    def _serialize_parameters(self, name: str, directory: Optional[str] = None) -> None:
        if not directory:
            directory = self.reports_dir
        filename = f'{name}.pickle'
        self.parameters.serialize_parameters_into(os.path.join(directory, filename))

    def _update_best_fit(self, task: Task) -> None:
        if not task.report['chi_squared']:
            return None
        current = task.report['chi_squared']
        best_so_far = self._best_fit.get('chi_squared', None)
        if not best_so_far or float(current) < float(best_so_far):
            self._best_fit = {
                'chi_squared': current,
                'name': f'{self.name}:{task.name}',
                'parameters': task.parameters.to_list(),
                'parameters_list': task.parameters.get_ordered_values(),
                'covariance_matrix': task.report.get('covariance_matrix'),
                'parameter_errors': task.report.get('parameter_errors'),
            }

    @abstractmethod
    def _create_task(self, task_name: str, task_class: type(Task)) -> Task:
        pass
