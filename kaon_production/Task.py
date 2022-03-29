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
        self.ts = ts
        self.ffs = ffs
        self.errors = errors
        self.t_0_isoscalar = t_0_isoscalar
        self.t_0_isovector = t_0_isovector
        self.reports_dir = reports_dir
        self.should_plot = plot

    def run(self):
        print(f'Running task {self.name}...')
        self._set_up()
        opt_params = self._fit()
        self.parameters.update_free_values(opt_params)
        self._report_fit(opt_params)
        self._plot(opt_params)
        return self.parameters

    def _fit(self):
        opt_params, _ = curve_fit(
            f=self.partial_f,
            xdata=self.ts,
            ydata=self.ffs,
            p0=self.parameters.get_free_values(),
            sigma=self.errors,
            absolute_sigma=False,
            bounds=self.parameters.get_bounds_for_free_parameters(),
            maxfev=5000,
        )
        return opt_params

    def _plot(self, opt_params):
        plot_ff_fit(self.ts, self.ffs, self.errors, self.partial_f,
                    opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)

    def _report_fit(self, opt_parameters):
        fit_ys = self.partial_f(self.ts, *opt_parameters)
        r_squared = [(data - fit) ** 2 for data, fit in zip(self.ffs, fit_ys)]
        chi_squared = math.sqrt(
            sum([r2 / (err ** 2) for r2, err in zip(r_squared, self.errors)])
        )
        report = {
            'name': self.name,
            'parameters': self.parameters.to_list(),
            'r2': sum(r_squared),
            'chi_squared': chi_squared,
        }
        print(f'Report: {report}\n')

    def _set_up(self):
        raise NotImplementedError
