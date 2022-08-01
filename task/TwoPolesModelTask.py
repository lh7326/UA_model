from typing import List

from plotting.plot_fit import plot_ff_fit_neutral_plus_charged
from task.Task import Task
from model_parameters.TwoPolesModelParameters import TwoPolesModelParameters

from kaon_production.data import Datapoint
from kaon_production.utils import make_partial_form_factor_for_parameters


class TwoPolesModelTask(Task):

    def __init__(self,
                 name: str,
                 parameters: TwoPolesModelParameters,
                 ts: List[Datapoint],
                 ys: List[float],
                 errors: List[float],
                 reports_dir: str,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ys, errors, reports_dir, plot, use_handpicked_bounds)

    def _plot(self, opt_params):
        plot_ff_fit_neutral_plus_charged(self.ts, self.ys, self.errors, self.partial_f,
                                         opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)

    def _set_up(self):
        self.partial_f = make_partial_form_factor_for_parameters(self.parameters)
