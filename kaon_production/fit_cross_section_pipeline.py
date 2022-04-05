from configparser import ConfigParser

from multiprocessing import Pool

from kaon_production.data import read_cross_section_data
from kaon_production.ModelParameters import ModelParameters
from kaon_production.tasks import (
    TaskFullFit, TaskFixedResonancesFit, TaskFixedCouplingConstants,
    TaskFitLowEnergies, TaskFitHighEnergies, TaskFitOnRandomSubsetOfData,
    TaskOnlyThresholdsFit, TaskFixedResonancesAndThresholdsFit)
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


def make_pipeline_fast(ts, cross_section_values, errors, k_meson_mass, alpha, hc_squared,
                       t_0_isoscalar, t_0_isovector, initial_params,
                       reports_dir, name='fast'):

    task_list = [
        TaskFixedResonancesFit, TaskFixedCouplingConstants, TaskFullFit,
    ]
    return Pipeline(name, initial_params, task_list,
                    ts, cross_section_values, errors, k_meson_mass, alpha, hc_squared,
                    t_0_isoscalar, t_0_isovector, reports_dir, plot=False)


def make_pipeline_medium(ts, cross_section_values, errors, k_meson_mass, alpha, hc_squared,
                         t_0_isoscalar, t_0_isovector, initial_params,
                         reports_dir, name='medium'):

    task_list = [
        TaskFixedResonancesFit,
        TaskFitLowEnergies,
        TaskFixedResonancesAndThresholdsFit,
        TaskOnlyThresholdsFit,
        TaskFixedCouplingConstants,
        TaskFullFit,
    ]
    return Pipeline(name, initial_params, task_list,
                    ts, cross_section_values, errors, k_meson_mass, alpha, hc_squared,
                    t_0_isoscalar, t_0_isovector, reports_dir, plot=False)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports'

    ts, cross_sections_values, errors = read_cross_section_data()

    def f(name):
        initial_parameters = perturb_model_parameters(
            make_initial_parameters(t_0_isoscalar, t_0_isovector),
            perturbation_size=0.3, perturbation_size_resonances=0.1,
        )
        pipeline = make_pipeline_fast(
            ts, cross_sections_values, errors, kaon_mass, alpha, hc_squared,
            t_0_isoscalar, t_0_isovector,
            initial_parameters, path_to_reports, name=name)
        return pipeline.run()

    final_results = []
    with Pool(processes=16) as pool:
        results = [pool.apply_async(f, (f'pool_fast_{i}',)) for i in range(300)]
        pool.close()
        pool.join()
        best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
        for result in results:
            r = result.get()
            print(r)
            if r and r.get('chi_squared', None) is not None:
                final_results.append(r)
                if best_fit['chi_squared'] is None:
                    best_fit = r
                elif r['chi_squared'] < best_fit['chi_squared']:
                    best_fit = r
        print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: fr['chi_squared'])[:10]:
        print(final_result)
