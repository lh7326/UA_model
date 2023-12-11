from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Union

from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint
from model_parameters import ModelParameters


class Task(ABC):

    def __init__(self,
                 name: str,
                 parameters: ModelParameters,
                 ts: Union[List[KaonDatapoint], List[NucleonDatapoint]],
                 ys: List[float],
                 errors: List[float],
                 plot: bool = True,
                 use_handpicked_bounds: bool = True):
        self.name = name
        self.parameters = parameters
        self.partial_f = None  # prepared in the _setup method
        self.ts = ts
        self.ys = ys
        self.errors = errors
        self.ts_fit = ts
        self.ys_fit = ys
        self.errors_fit = errors
        self.should_plot = plot
        self.use_handpicked_bounds = use_handpicked_bounds
        self.report = {
            'name': self.name,
            'initial_parameters': self.parameters.to_list(),
            'final_parameters': None,
            'r2': None,
            'chi_squared': None,
            'covariance_matrix': None,
            'parameter_errors': None,
            'status': 'started',
            'error_message': None,
            'parameter_list': [],
        }

    def run(self):
        self._set_up()
        opt_params, covariance_matrix = self._fit()
        if opt_params is not None:  # opt_params are None if the fit ends in runtime error
            self.parameters.update_free_values(opt_params)  # type: ignore
            self._update_report(opt_params, covariance_matrix)
            self._plot(opt_params)
        return self.parameters

    def _fit(self):
        try:
            opt_params, covariance_matrix = curve_fit(
                f=self.partial_f,
                xdata=self.ts_fit,
                ydata=self.ys_fit,
                p0=self.parameters.get_free_values(),
                sigma=self.errors_fit,
                absolute_sigma=True,
                bounds=self.parameters.get_bounds_for_free_parameters(handpicked=self.use_handpicked_bounds),
                maxfev=7000,
            )
        except RuntimeError as err:
            self.report['status'] = 'failed'
            self.report['error_message'] = str(err)  # type: ignore
            opt_params = None
            covariance_matrix = None
        return opt_params, covariance_matrix

    def _update_report(self, opt_parameters, covariance_matrix):
        fit_ys = self.partial_f(self.ts, *opt_parameters)
        r_squared = [(data - fit) ** 2 for data, fit in zip(self.ys, fit_ys)]
        errors = self.errors
        chi_squared = (
            sum([r2 / (err ** 2) for r2, err in zip(r_squared, errors)])
        ) / (len(r_squared) - len(opt_parameters))

        self.report.update(
            final_parameters=self.parameters.to_list(),
            r2=sum(r_squared),
            chi_squared=chi_squared,
            covariance_matrix=covariance_matrix,
            parameter_errors=np.sqrt(np.diag(covariance_matrix)),
            status='finished',
            parameter_list=self.parameters.get_ordered_values(),
        )

    @abstractmethod
    def _plot(self, opt_params):
        pass

    @abstractmethod
    def _set_up(self):
        pass
