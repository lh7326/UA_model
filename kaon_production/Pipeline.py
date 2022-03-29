from typing import List, Type

from kaon_production.ModelParameters import ModelParameters
from kaon_production.Task import Task


class Pipeline:

    def __init__(self, name: str, parameters: ModelParameters, tasks: List[Type[Task]],
                 t_values: List[float], form_factors: List[float], errors: List[float],
                 t_0_isoscalar: float, t_0_isovector: float, plot: bool = True):
        self.name = name
        self.parameters = parameters
        self.tasks = tasks
        self.t_values = t_values
        self.form_factors = form_factors
        self.errors = errors
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self.plot = plot

    def run(self):
        res = None
        self._log(f'Starting. Initial parameters: {self.parameters.to_list()}')
        for i, task_class in enumerate(self.tasks):
            self._log(f'Initializing task #{i}. Parameters: {self.parameters.to_list()}')
            task_name = f'Task#{i}:{task_class.__name__}'
            task = task_class(
                task_name, self.parameters, self.t_values, self.form_factors, self.errors,
                self.t_0_isoscalar, self.t_0_isovector, self.plot
            )

            self._log(f'Running {task_name}')
            res = task.run()
            self._log(f'{task_name} final parameters: {task.parameters.to_list()}')

            self.parameters = task.parameters
            self.parameters.release_all_parameters()
        return res

    def _log(self, msg):
        print(f'Pipeline {self.name}: {msg}')
