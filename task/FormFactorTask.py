from abc import ABC
from typing import List, Union

from kaon_production.data import Datapoint
from plotting.plot_fit import plot_ff_fit_neutral_plus_charged
from model_parameters import KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected
from task.Task import Task


class FormFactorTask(Task, ABC):

    def __init__(self,
                 name: str,
                 parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedSelected],
                 ts: List[Datapoint],
                 ffs: List[float],
                 errors: List[float],
                 reports_dir: str,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ffs, errors, reports_dir, plot, use_handpicked_bounds)

    def _plot(self, opt_params):
        plot_ff_fit_neutral_plus_charged(self.ts, self.ys, self.errors, self.partial_f,
                                         opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)
