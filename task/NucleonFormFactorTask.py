from abc import ABC
from typing import List, Optional

from nucleon_production.data import NucleonDatapoint
from plotting.plot_fit import plot_ff_fit_electric_plus_magnetic
from model_parameters import NucleonParameters
from task.Task import Task


class NucleonFormFactorTask(Task, ABC):

    def __init__(self,
                 name: str,
                 parameters: NucleonParameters,
                 ts: List[NucleonDatapoint],
                 ffs: List[float],
                 errors: List[float],
                 reports_dir: Optional[str] = None,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ffs, errors, plot, use_handpicked_bounds)
        self.reports_dir = reports_dir

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_ff_fit_electric_plus_magnetic(self.ts, self.ys, self.errors, self.partial_f,
                                               opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)
