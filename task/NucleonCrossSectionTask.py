from abc import ABC
from typing import List, Union, Optional

from nucleon_production.data import NucleonDatapoint
from plotting.plot_fit import plot_cs_fit
from model_parameters import NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters
from task.Task import Task


class NucleonCrossSectionTask(Task, ABC):

    def __init__(self,
                 name: str,
                 parameters: Union[NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters],
                 ts: List[NucleonDatapoint],
                 css: List[float],
                 errors: List[float],
                 product_particle_mass: float,
                 alpha: float,
                 hc_squared: float,
                 reports_dir: Optional[str] = None,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, css, errors, plot, use_handpicked_bounds)
        self.reports_dir = reports_dir
        self.product_particle_mass = product_particle_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_cs_fit(self.ts, self.ys, self.errors, self.partial_f,
                        opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)
