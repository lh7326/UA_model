from abc import ABC
from typing import List, Union, Optional

from kaon_production.data import KaonDatapoint
from plotting.plot_fit import plot_combined_fit
from model_parameters import KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedSelected
from task.Task import Task


class KaonCombinedTask(Task, ABC):

    def __init__(self,
                 name: str,
                 parameters: Union[KaonParameters, KaonParametersB,
                                   KaonParametersSimplified, KaonParametersFixedSelected],
                 ts: List[KaonDatapoint],
                 ys: List[float],
                 errors: List[float],
                 product_particle_mass: float,
                 alpha: float,
                 hc_squared: float,
                 reports_dir: Optional[str] = None,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ys, errors, plot, use_handpicked_bounds)
        self.reports_dir = reports_dir
        self.product_particle_mass = product_particle_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_combined_fit(self.ts, self.ys, self.errors, self.partial_f,
                              opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)