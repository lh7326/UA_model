from kaon_production.data import read_data, Datapoint
from model_parameters import ETGMRModelParameters, TwoPolesModelParameters
from pipeline.FormFactorPipeline import FormFactorPipeline
from task.ETGMRModelTask import ETGMRModelTask
from task.TwoPolesModelTask import TwoPolesModelTask
from task.ResidualOscillationsTask import ResidualOscillationsTask
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
    #charged_ts = [Datapoint(t=t, is_charged=True) for t in charged_ts]

    def make_pipeline(
            ts_charged, ys_charged, errors_charged,
            ts_neutral, ys_neutral, errors_neutral,
            initial_params, reports_dir, name='pipeline', handpicked=False, model='etgmr'):

        task_list = []
        if model.lower() == 'etgmr':
            task_list.append(ETGMRModelTask)
        elif model.lower() == 'twopoles':
            task_list.append(TwoPolesModelTask)
        else:
            raise 'Unknown model'

        task_list.append(ResidualOscillationsTask)

        return FormFactorPipeline(
            name, initial_params, task_list,
            ts_charged, ys_charged, errors_charged,
            ts_neutral, ys_neutral, errors_neutral,
            reports_dir, plot=True, use_handpicked_bounds=handpicked)

    def f(name, model='etgmr'):
        initial_parameters = make_initial_parameters(model=model)

        initial_parameters = perturb_model_parameters(
            initial_parameters,
            perturbation_size=1.0, perturbation_size_resonances=0.1,
            use_handpicked_bounds=False,
        )

        pipeline = make_pipeline(charged_ts, charged_ff_values, charged_errors, [], [], [],
                                 initial_parameters, path_to_reports, f'{model}_{name}',
                                 handpicked=False, model=model)

        return pipeline.run()

    final_results = []
    best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
    for i in range(1):
        result = f(f'_1_{i}', model='etgmr')
        print(result)
        if result and result.get('chi_squared', None) is not None:
            final_results.append(result)
            if best_fit['chi_squared'] is None or result['chi_squared'] < best_fit['chi_squared']:
                best_fit = result
    print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: fr['chi_squared'])[:3]:
        print(final_result)
