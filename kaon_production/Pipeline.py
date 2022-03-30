from typing import List, Type
import os.path

from kaon_production.ModelParameters import ModelParameters
from kaon_production.Task import Task


class Pipeline:

    def __init__(self, name: str, parameters: ModelParameters, tasks: List[Type[Task]],
                 t_values: List[float], form_factors: List[float], errors: List[float],
                 t_0_isoscalar: float, t_0_isovector: float, reports_dir: str, plot: bool = True):
        self.name = name
        self.parameters = parameters
        self.tasks = tasks
        self.t_values = t_values
        self.form_factors = form_factors
        self.errors = errors
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self.reports_dir = os.path.join(reports_dir, name)
        self.plot = plot

        self._report = f'Report {name}:\n'
        self._set_up_reports_directory()

    def run(self):
        res = None
        self._log(f'Starting. Initial parameters: {self.parameters.to_list()}')
        for i, task_class in enumerate(self.tasks):
            self._log(f'Initializing Task#{i}. Parameters: {self.parameters.to_list()}')
            task_name = f'Task#{i}:{task_class.__name__}'
            task = task_class(
                task_name, self.parameters, self.t_values, self.form_factors, self.errors,
                self.t_0_isoscalar, self.t_0_isovector, self.reports_dir, self.plot
            )

            self._log(f'Running {task_name}')
            res = task.run()
            self._log(f'{task_name} report: {task.report}')

            self.parameters = task.parameters
            self.parameters.release_all_parameters()
            self._flush_report()
        return res

    def _log(self, msg: str):
        print(f'Pipeline {self.name}: {msg}\n')
        self._report += (msg + '\n')

    def _set_up_reports_directory(self):
        #TODO: do better!
        if os.path.exists(self.reports_dir):
            raise ValueError(f'Directory already exists: {self.reports_dir}')
        os.mkdir(self.reports_dir)

    def _flush_report(self):
        filepath = os.path.join(self.reports_dir,'report.txt')
        with open(filepath, 'a') as f:
            f.write(self._report)
        self._report = '\n'
