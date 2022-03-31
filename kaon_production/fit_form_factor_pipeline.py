from configparser import ConfigParser

from multiprocessing import Pool

from kaon_production.data import read_form_factor_data
from kaon_production.ModelParameters import ModelParameters
from kaon_production.tasks import (
    TaskFullFit, TaskFixedResonancesFit, TaskFixedCouplingConstants,
    TaskFitLowEnergies, TaskFitHighEnergies, TaskFitOnRandomSubsetOfData)
from kaon_production.Pipeline import Pipeline
from kaon_production.utils import perturb_model_parameters


def make_initial_parameters(t_0_isoscalar, t_0_isovector):
    return ModelParameters(
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=1.0,
        t_in_isovector=1.0,
        a_omega=1.0/12,
        mass_omega=0.78266,
        decay_rate_omega=0.00868,
        a_omega_prime=1.0/12,
        mass_omega_prime=1.410,
        decay_rate_omega_prime=0.29,
        a_omega_double_prime=1.0/12,
        mass_omega_double_prime=1.670,
        decay_rate_omega_double_prime=0.315,
        a_phi=1.0/12,
        mass_phi=1.019461,
        decay_rate_phi=0.004249,
        a_phi_prime=1.0/12,
        mass_phi_prime=1.680,
        decay_rate_phi_prime=0.150,
        mass_phi_double_prime=2.159,
        decay_rate_phi_double_prime=0.137,
        a_rho=1.0/8,
        mass_rho=0.77525,
        decay_rate_rho=0.1474,
        a_rho_prime=1.0/8,
        mass_rho_prime=1.465,
        decay_rate_rho_prime=0.4,
        a_rho_double_prime=1.0/8,
        mass_rho_double_prime=1.720,
        decay_rate_rho_double_prime=0.25,
        mass_rho_triple_prime=2.15,
        decay_rate_rho_triple_prime=0.3,
    )


def make_pipeline_fast(ts, form_factors_values, errors,
                       t_0_isoscalar, t_0_isovector, initial_params,
                       reports_dir, name='fast'):

    task_list = [
        TaskFixedResonancesFit, TaskFixedCouplingConstants, TaskFullFit,
    ]
    return Pipeline(name, initial_params, task_list,
                    ts, form_factors_values, errors,
                    t_0_isoscalar, t_0_isovector, reports_dir, plot=False)


def make_pipeline_medium(ts, form_factors_values, errors,
                         t_0_isoscalar, t_0_isovector, initial_params,
                         reports_dir, name='medium'):

    task_list = [
        TaskFixedResonancesFit, TaskFitLowEnergies, TaskFixedResonancesFit,
        TaskFixedResonancesFit, TaskFitOnRandomSubsetOfData,
        TaskFixedResonancesFit, TaskFitOnRandomSubsetOfData,
        TaskFixedCouplingConstants, TaskFullFit
    ]
    return Pipeline(name, initial_params, task_list,
                    ts, form_factors_values, errors,
                    t_0_isoscalar, t_0_isovector, reports_dir, plot=False)


def make_pipeline3(ts, form_factors_values, errors,
                   t_0_isoscalar, t_0_isovector, initial_params,
                   reports_dir, name='Pipeline3'):
    task_list = [
        TaskFixedResonancesFit, TaskFixedCouplingConstants,
        TaskFitLowEnergies, TaskFitHighEnergies,
        TaskFixedResonancesFit, TaskFixedCouplingConstants,
        TaskFitOnRandomSubsetOfData, TaskFitOnRandomSubsetOfData,
        TaskFixedResonancesFit, TaskFitOnRandomSubsetOfData,
        TaskFixedResonancesFit, TaskFitOnRandomSubsetOfData,
        TaskFixedResonancesFit, TaskFullFit,
    ]
    return Pipeline(name, initial_params, task_list,
                    ts, form_factors_values, errors,
                    t_0_isoscalar, t_0_isovector, reports_dir, plot=False)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    path_to_reports = '/home/lukas/reports'

    ts, form_factors_values, errors = read_form_factor_data()

    def f(name):
        initial_parameters = perturb_model_parameters(
            make_initial_parameters(t_0_isoscalar, t_0_isovector)
        )
        pipeline = make_pipeline_fast(
            ts, form_factors_values, errors, t_0_isoscalar, t_0_isovector,
            initial_parameters, path_to_reports, name=name)
        return pipeline.run()

    with Pool(processes=2) as pool:
        results = [pool.apply_async(f, (f'pool_fast_{i}',)) for i in range(100)]
        pool.close()
        pool.join()
        best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
        for result in results:
            r = result.get()
            print(r)
            if r and r.get('chi_squared', None) is not None:
                if best_fit['chi_squared'] is None:
                    best_fit = r
                elif r['chi_squared'] < best_fit['chi_squared']:
                    best_fit = r
        print('Best fit: ', best_fit)
