from abc import ABC
from typing import List, Union, Optional

from kaon_production.data import KaonDatapoint
from plotting.plot_fit import plot_ff_fit_neutral_plus_charged
from model_parameters import KaonParameters, KaonParametersB, KaonParametersSimplified, KaonParametersFixedSelected
from task.Task import Task


class KaonFormFactorTask(Task, ABC):

    def __init__(self,
                 name: str,
                 parameters: Union[KaonParameters, KaonParametersB, KaonParametersSimplified,
                                   KaonParametersFixedSelected],
                 ts: List[KaonDatapoint],
                 ffs: List[float],
                 errors: List[float],
                 reports_dir: Optional[str] = None,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ffs, errors, plot, use_handpicked_bounds)
        self.reports_dir = reports_dir

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_ff_fit_neutral_plus_charged(self.ts, self.ys, self.errors, self.partial_f,
                                             opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)
