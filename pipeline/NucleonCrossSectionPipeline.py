from typing import List, Tuple, Type, Union

from nucleon_production.data import NucleonDatapoint
from model_parameters import NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters
from pipeline.Pipeline import Pipeline
from task.NucleonCrossSectionTask import NucleonCrossSectionTask


class NucleonCrossSectionPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters],
                 tasks: List[Type[NucleonCrossSectionTask]],
                 t_values_proton_electric: List[float], cross_sections_proton_electric: List[float],
                 errors_proton_electric: List[float],
                 t_values_proton_magnetic: List[float], cross_sections_proton_magnetic: List[float],
                 errors_proton_magnetic: List[float],
                 t_values_neutron_electric: List[float], cross_sections_neutron_electric: List[float],
                 errors_neutron_electric: List[float],
                 t_values_neutron_magnetic: List[float], cross_sections_neutron_magnetic: List[float],
                 errors_neutron_magnetic: List[float],
                 nucleon_mass: float, alpha: float, hc_squared: float, reports_dir: str,
                 plot: bool = True, use_handpicked_bounds: bool = True) -> None:
        ts, ys, errors = self._prepare_data(
            t_values_proton_electric, cross_sections_proton_electric, errors_proton_electric,
            t_values_proton_magnetic, cross_sections_proton_magnetic, errors_proton_magnetic,
            t_values_neutron_electric, cross_sections_neutron_electric, errors_neutron_electric,
            t_values_neutron_magnetic, cross_sections_neutron_magnetic, errors_neutron_magnetic,
        )

        super().__init__(name, parameters, tasks,
                         ts, ys, errors,
                         reports_dir, plot, use_handpicked_bounds)
        self.nucleon_mass = nucleon_mass
        self.alpha = alpha
        self.hc_squared = hc_squared

    def _create_task(self, task_name: str, task_class: type(NucleonCrossSectionTask)) -> NucleonCrossSectionTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.nucleon_mass, self.alpha, self.hc_squared,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

    @staticmethod
    def _prepare_data(
            ts_proton_electric: List[float], css_proton_electric: List[float], errors_proton_electric: List[float],
            ts_proton_magnetic: List[float], css_proton_magnetic: List[float], errors_proton_magnetic: List[float],
            ts_neutron_electric: List[float], css_neutron_electric: List[float], errors_neutron_electric: List[float],
            ts_neutron_magnetic: List[float], css_neutron_magnetic: List[float], errors_neutron_magnetic: List[float],
    ) -> Tuple[List[NucleonDatapoint], List[float], List[float]]:
        ts = [NucleonDatapoint(t, True, True) for t in ts_proton_electric]
        ys = list(css_proton_electric)
        errors = list(errors_proton_electric)
        ts += [NucleonDatapoint(t, True, False) for t in ts_proton_magnetic]
        ys += list(css_proton_magnetic)
        errors += list(errors_proton_magnetic)
        ts += [NucleonDatapoint(t, False, True) for t in ts_neutron_electric]
        ys += list(css_neutron_electric)
        errors += list(errors_neutron_electric)
        ts += [NucleonDatapoint(t, False, False) for t in ts_neutron_magnetic]
        ys += list(css_neutron_magnetic)
        errors += list(errors_neutron_magnetic)

        ts, ys, errors = zip(
            *sorted(
                zip(ts, ys, errors),
                key=lambda tup: tup[0].t,
            )
        )
        return ts, ys, errors
