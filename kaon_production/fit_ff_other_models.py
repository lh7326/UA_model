from configparser import ConfigParser

from kaon_production.data import read_data, Datapoint
from model_parameters import ETGMRModelParameters, TwoPolesModelParameters
from task.ETGMRModelTask import ETGMRModelTask
from task.TwoPolesModelTask import TwoPolesModelTask
from kaon_production.utils import perturb_model_parameters


def make_initial_parameters(model: str='etgmr'):
    if model.lower() == 'etgmr':
        return ETGMRModelParameters(
            a=1.0,
            m_a=15.0,
        )
    elif model.lower() == 'twopoles':
        return TwoPolesModelParameters(
            a=1.0,
            m_1=0.5,
            m_2=2.0,
        )
    else:
        raise 'Unknown model type'


if __name__ == '__main__':
    path_to_reports = '/home/lukas/reports/ff'

    charged_ts, charged_ff_values, charged_errors = read_data('charged_ff_2.csv')
    charged_ts = [Datapoint(t=t, is_charged=True) for t in charged_ts]

    def f(name, model='etgmr'):
        initial_parameters = make_initial_parameters(model=model)

        initial_parameters = perturb_model_parameters(
            initial_parameters,
            perturbation_size=1.0, perturbation_size_resonances=0.1,
            use_handpicked_bounds=False,
        )

        if isinstance(initial_parameters, ETGMRModelParameters):
            task = ETGMRModelTask(name=f'ETGR_{name}', parameters=initial_parameters,
                                  ts=charged_ts, ys=charged_ff_values, errors=charged_errors,
                                  reports_dir=path_to_reports, plot=True, use_handpicked_bounds=False)
        elif isinstance(initial_parameters, TwoPolesModelParameters):
            task = TwoPolesModelTask(name=f'TwoPoles_{name}', parameters=initial_parameters,
                                     ts=charged_ts, ys=charged_ff_values, errors=charged_errors,
                                     reports_dir=path_to_reports, plot=True, use_handpicked_bounds=False)
        else:
            raise 'Unknown parameters type'

        task.run()
        return {
                'chi_squared': task.report['chi_squared'],
                'name': task.name,
                'parameters': task.parameters.to_list(),
                'covariance_matrix': task.report.get('covariance_matrix'),
                'parameter_errors': task.report.get('parameter_errors'),
            }

    final_results = []
    best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
    for i in range(3):
        result = f(f'_1_{i}', model='twopoles')
        print(result)
        if result and result.get('chi_squared', None) is not None:
            final_results.append(result)
            if best_fit['chi_squared'] is None or result['chi_squared'] < best_fit['chi_squared']:
                best_fit = result
    print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: fr['chi_squared'])[:3]:
        print(final_result)
