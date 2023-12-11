import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from plotting.plot_fit import plot_background_residuals
from task.Task import Task
from model_parameters import ModelParameters, Parameter
from other_models import DampedOscillations

from kaon_production.data import KaonDatapoint
from nucleon_production.data import NucleonDatapoint
from common.utils import make_partial_cross_section_for_parameters


class _OscillationsParameters(ModelParameters):

    def __init__(self, a: float, b: float, c: float, d: float) -> None:

        super().__init__(a, b, c, d, always_fixed=())

    @staticmethod
    def _setup_data(a: float, b: float, c: float, d: float) -> List[Parameter]:

        return [
            Parameter(name='a', value=a, is_fixed=False),
            Parameter(name='b', value=b, is_fixed=False),
            Parameter(name='c', value=c, is_fixed=False),
            Parameter(name='d', value=d, is_fixed=False),
        ]

    @classmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> '_OscillationsParameters':
        kwargs = {par.name: par.value for par in list_of_parameters}
        instance = cls(**kwargs)
        parameters_to_fix = [p.name for p in list_of_parameters if p.is_fixed]
        instance.release_all_parameters()
        instance.fix_parameters(parameters_to_fix)
        return instance

    def get_bounds_for_free_parameters(self, handpicked: bool = True) -> Tuple[List[float], List[float]]:
        if handpicked:
            full_bounds = self.get_model_parameters_bounds_handpicked()
        else:
            full_bounds = self.get_model_parameters_bounds_maximal()
        lower_bounds = []
        upper_bounds = []
        for parameter in self._data:
            if not parameter.is_fixed:
                bounds = full_bounds[parameter.name]
                lower_bounds.append(bounds['lower'])
                upper_bounds.append(bounds['upper'])
        return lower_bounds, upper_bounds

    @staticmethod
    def get_model_parameters_bounds_handpicked() -> Dict:
        """
        Returns a handpicked set of bounds.

        """
        return {
            'a': {'lower': -np.inf, 'upper': np.inf},
            'b': {'lower': 0.0, 'upper': np.inf},
            'c': {'lower': 0.0, 'upper': np.inf},
            'd': {'lower': -np.inf, 'upper': np.inf},
        }

    @staticmethod
    def get_model_parameters_bounds_maximal() -> Dict:
        return {
            'a': {'lower': -np.inf, 'upper': np.inf},
            'b': {'lower': -np.inf, 'upper': np.inf},
            'c': {'lower': -np.inf, 'upper': np.inf},
            'd': {'lower': -np.inf, 'upper': np.inf},
        }

    def get_ordered_values(self):
        return [self['a'].value, self['b'].value, self['c'].value, self['d'].value]


class ResidualOscillationsTask(Task):
    # NOTE: Adjusted for the needs of nucleon oscillations work.

    def __init__(self,
                 name: str,
                 parameters: ModelParameters,
                 ts: Union[List[KaonDatapoint], List[NucleonDatapoint]],
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
        self.eff_ffs = None
        self.background_fit = None
        self.ff_errors = None
        self.ff_ts = None

    def _plot(self, opt_params):
        if self.reports_dir:
            plot_background_residuals(
                self.ff_ts, self.eff_ffs, self.background_fit, self.ff_errors,
                self.ts, self.ys, self.errors, self.partial_f,
                opt_params, self.name, show=self.should_plot, save_dir=self.reports_dir)

    def transform_to_effective_form_factor(self, cs, err, t):
        tau = t / (4 * (self.product_particle_mass**2))
        beta = math.sqrt(1 - 1/tau)
        denominator = 2 * math.pi * (self.alpha**2) * beta * (2 + 1/tau)
        ff_squared = 3 * t * cs / denominator
        error_for_ff_squared = 3 * t * err / denominator
        ff = math.sqrt(ff_squared)
        return ff, 0.5 * error_for_ff_squared / ff

    def _set_up(self):
        # recover the background function
        self.parameters.fix_all_parameters()
        background_f = make_partial_cross_section_for_parameters(
            self.product_particle_mass, self.alpha, self.hc_squared, self.parameters)
        background_ys = background_f(self.ts)

        effective_ff_fit, effective_errs = zip(
            *[self.transform_to_effective_form_factor(cs, err, datapoint.t)
              for cs, err, datapoint
              in zip(background_ys, self.errors, self.ts)]
        )
        effective_ff_data, _ = zip(
            *[self.transform_to_effective_form_factor(cs, err, datapoint.t)
              for cs, err, datapoint
              in zip(self.ys, self.errors, self.ts)]
        )
        self.ff_ts = self.ts
        self.eff_ffs = effective_ff_data
        self.background_fit = effective_ff_fit
        self.ys = self.ys_fit = [y - b for y, b in zip(effective_ff_data, effective_ff_fit)]
        self.ff_errors = self.errors_fit = self.errors = effective_errs

        self.map_ts_to_laboratory_system_momentum(self.product_particle_mass)
        self.drop_large_momenta(5)

        self.parameters = _OscillationsParameters(a=100.0, b=2.0, c=5.0, d=0.0)

        def partial(ts, *pars):
            parameters = _OscillationsParameters.from_list(self.parameters.to_list())
            parameters.update_free_values(list(pars))
            model = DampedOscillations(
                a=parameters['a'].value,
                b=parameters['b'].value,
                c=parameters['c'].value,
                d=parameters['d'].value,
            )
            results = []
            for datapoint in ts:
                if isinstance(datapoint, KaonDatapoint):
                    results.append(model(datapoint.t))
                else:
                    results.append(model(float(datapoint[0])))

            return results
        self.partial_f = partial

    def map_ts_to_laboratory_system_momentum(self, m):
        if not self.ts:
            return

        if isinstance(self.ts[0], KaonDatapoint):
            ps = [
                KaonDatapoint(t=math.sqrt(d.t * (d.t - 4 * (m ** 2))) / (2 * m), is_charged=d.is_charged)
                for d in self.ts
            ]
        elif isinstance(self.ts[0], NucleonDatapoint):
            ps = [
                NucleonDatapoint(t=math.sqrt(d.t * (d.t - 4 * (m ** 2))) / (2 * m),
                                 proton=d.proton, electric=d.electric)
                for d in self.ts
            ]
        else:
            raise f'Bad type of t: {type(self.ts[0])}'

        self.ts_fit = self.ts = ps

    def drop_large_momenta(self, cutoff):
        threshold = cutoff
        nonrelativistic_momenta, vals, errs = zip(*[
            (p, v, e) for p, v, e in zip(self.ts, self.ys, self.errors) if p.t < threshold])
        self.ts_fit = self.ts = nonrelativistic_momenta
        self.ys_fit = self.ys = vals
        self.errors_fit = self.errors = errs
