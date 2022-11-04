from typing import List, Tuple, Type, Union

from model_parameters import NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters
from nucleon_production.data import NucleonDatapoint
from pipeline.Pipeline import Pipeline
from task.NucleonFormFactorTask import NucleonFormFactorTask


class NucleonFormFactorPipeline(Pipeline):

    def __init__(self, name: str,
                 parameters: Union[NucleonParameters, ETGMRModelParameters, TwoPolesModelParameters],
                 tasks: List[Type[NucleonFormFactorTask]],
                 t_values_proton_electric: List[float], form_factors_proton_electric: List[float],
                 errors_proton_electric: List[float],
                 t_values_proton_magnetic: List[float], form_factors_proton_magnetic: List[float],
                 errors_proton_magnetic: List[float],
                 t_values_neutron_electric: List[float], form_factors_neutron_electric: List[float],
                 errors_neutron_electric: List[float],
                 t_values_neutron_magnetic: List[float], form_factors_neutron_magnetic: List[float],
                 errors_neutron_magnetic: List[float],
                 reports_dir: str, plot: bool = True, use_handpicked_bounds: bool = True) -> None:

        ts, ys, errors = self._prepare_data(
            t_values_proton_electric, form_factors_proton_electric, errors_proton_electric,
            t_values_proton_magnetic, form_factors_proton_magnetic, errors_proton_magnetic,
            t_values_neutron_electric, form_factors_neutron_electric, errors_neutron_electric,
            t_values_neutron_magnetic, form_factors_neutron_magnetic, errors_neutron_magnetic,
        )
        super().__init__(name, parameters, tasks,
                         ts, ys, errors,
                         reports_dir, plot, use_handpicked_bounds)

    def _create_task(self, task_name: str, task_class: type(NucleonFormFactorTask)) -> NucleonFormFactorTask:
        return task_class(
            task_name, self.parameters,
            self.ts, self.ys, self.errors,
            self.reports_dir, self.plot, self.use_handpicked_bounds
        )

    @staticmethod
    def _prepare_data(
            ts_proton_electric: List[float], ffs_proton_electric: List[float], errors_proton_electric: List[float],
            ts_proton_magnetic: List[float], ffs_proton_magnetic: List[float], errors_proton_magnetic: List[float],
            ts_neutron_electric: List[float], ffs_neutron_electric: List[float], errors_neutron_electric: List[float],
            ts_neutron_magnetic: List[float], ffs_neutron_magnetic: List[float], errors_neutron_magnetic: List[float],
    ) -> Tuple[List[NucleonDatapoint], List[float], List[float]]:
        ts = [NucleonDatapoint(t, True, True) for t in ts_proton_electric]
        ys = list(ffs_proton_electric)
        errors = list(errors_proton_electric)
        ts += [NucleonDatapoint(t, True, False) for t in ts_proton_magnetic]
        ys += list(ffs_proton_magnetic)
        errors += list(errors_proton_magnetic)
        ts += [NucleonDatapoint(t, False, True) for t in ts_neutron_electric]
        ys += list(ffs_neutron_electric)
        errors += list(errors_neutron_electric)
        ts += [NucleonDatapoint(t, False, False) for t in ts_neutron_magnetic]
        ys += list(ffs_neutron_magnetic)
        errors += list(errors_neutron_magnetic)

        ts, ys, errors = zip(
            *sorted(
                zip(ts, ys, errors),
                key=lambda tup: tup[0].t,
            )
        )
        return ts, ys, errors
