import math
from scipy.optimize import curve_fit

from plotting.plot_fit import plot_ff_fit
from kaon_production.ModelParameters import ModelParameters


class Task:

    def __init__(self, name: str, parameters: ModelParameters, ts, ffs, errors,
                 t_0_isoscalar, t_0_isovector, reports_dir, plot=True):
        self.name = name
        self.parameters = parameters
        self.partial_f = None  # prepared in the _setup method
        self.ts_report = ts
        self.ffs_report = ffs
        self.errors_report = errors
        self.ts_fit = ts
        self.ffs_fit = ffs
        self.errors_fit = errors
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self.reports_dir = reports_dir
        self.should_plot = plot
        self.report = None

    def run(self):
        self._set_up()
        opt_params = self._fit()
        if opt_params is not None:  # opt_params are None if the fit ends in runtime error
            self.parameters.update_free_values(opt_params)
            self._build_report(opt_params)
            self._plot(opt_params)
        return self.parameters

    def _fit(self):
        try:
            opt_params, _ = curve_fit(
                f=self.partial_f,
                xdata=self.ts_fit,
                ydata=self.ffs_fit,
                p0=self.parameters.get_free_values(),
                sigma=self.errors_fit,
                absolute_sigma=False,
                bounds=self.parameters.get_bounds_for_free_parameters(),
                maxfev=15000,
            )
        except RuntimeError as err:
            self.report = f'Task {self.name} failed: {err}'
            opt_params = None
        return opt_params

    def _plot(self, opt_params):
        plot_ff_fit(self.ts_report, self.ffs_report, self.errors_report, self.partial_f,
                    opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)

    def _build_report(self, opt_parameters):
        fit_ys = self.partial_f(self.ts_report, *opt_parameters)
        r_squared = [(data - fit) ** 2 for data, fit in zip(self.ffs_report, fit_ys)]
        chi_squared = (
            sum([r2 / (err ** 2) for r2, err in zip(r_squared, self.errors_report)])
        ) / len(self.ts_report)
        sqrt_sum_chi_squared = math.sqrt(
            sum([r2 / (err ** 2) for r2, err in zip(r_squared, self.errors_report)])
        )
        self.report = {
            'name': self.name,
            'parameters': self.parameters.to_list(),
            'r2': sum(r_squared),
            'chi_squared': chi_squared,
            'sqrt_sum_chi_squared': sqrt_sum_chi_squared,
        }

    def _set_up(self):
        raise NotImplementedError
