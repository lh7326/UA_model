from typing import List, Union, Optional

from plotting.plot_fit import plot_ff_fit_neutral_plus_charged
from task.Task import Task
from model_parameters.TwoPolesModelParameters import TwoPolesModelParameters

from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint
from common.utils import make_partial_cross_section_for_parameters


class TwoPolesModelTask(Task):

    def __init__(self,
                 name: str,
                 parameters: TwoPolesModelParameters,
                 ts: Union[List[KaonDatapoint], List[NucleonDatapoint]],
                 ys: List[float],
                 errors: List[float],
                 reports_dir: Optional[str] = None,
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        super().__init__(name, parameters, ts, ys, errors, plot, use_handpicked_bounds)
        self.reports_dir = reports_dir

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_ff_fit_neutral_plus_charged(self.ts, self.ys, self.errors, self.partial_f,
                                             opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)

    def _set_up(self):
        self.partial_f = make_partial_cross_section_for_parameters(0, 0, 0, self.parameters)
