from configparser import ConfigParser

from multiprocessing import Pool

from kaon_production.data import read_cross_section_data
from model_parameters import KaonParametersSimplified
from task.cross_section_tasks import TaskFixedResonancesFit
from pipeline.CrossSectionPipeline import CrossSectionPipeline
from kaon_production.utils import perturb_model_parameters


def make_initial_parameters(t_0_isoscalar, t_0_isovector):
    return KaonParametersSimplified(
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=1.0,
        t_in_isovector=1.0,
        a_omega_prime=1.0/10,
        mass_omega_prime=1.410,
        decay_rate_omega_prime=0.29,
        a_omega_double_prime=1.0/10,
        mass_omega_double_prime=1.670,
        decay_rate_omega_double_prime=0.315,
        a_phi=1.0/10,
        mass_phi=1.019461,
        decay_rate_phi=0.004249,
        a_phi_prime=1.0/10,
        mass_phi_prime=1.680,
        decay_rate_phi_prime=0.150,
        mass_phi_double_prime=2.159,
        decay_rate_phi_double_prime=0.137,
        a_rho_prime=1.0/6,
        mass_rho_prime=1.34231,
        decay_rate_rho_prime=0.49217,
        a_rho_double_prime=1.0/6,
        mass_rho_double_prime=1.7185,
        decay_rate_rho_double_prime=0.48958,
        mass_rho_triple_prime=2.15,
        decay_rate_rho_triple_prime=0.3,
    )


def make_pipeline_restricted(
        ts_charged, cross_section_values_charged, errors_charged,
        ts_neutral, cross_section_values_neutral, errors_neutral,
        k_meson_mass, alpha, hc_squared, initial_params,
        reports_dir, name='restricted', handpicked=False):

    task_list = [
        TaskFixedResonancesFit,
    ]
    return CrossSectionPipeline(name, initial_params, task_list,
                                ts_charged, cross_section_values_charged, errors_charged,
                                ts_neutral, cross_section_values_neutral, errors_neutral,
                                k_meson_mass, alpha, hc_squared,
                                reports_dir, plot=False, use_handpicked_bounds=handpicked)


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

    charged_ts, charged_cross_sections_values, charged_errors = read_cross_section_data('charged_kaon.csv')
    neutral_ts, neutral_cross_sections_values, neutral_errors = read_cross_section_data('neutral_kaon.csv')

    def f(name):
        initial_parameters = make_initial_parameters(t_0_isoscalar, t_0_isovector)
        initial_parameters.fix_parameters([
            'mass_omega_prime', 'decay_rate_omega_prime',
            'mass_phi', 'decay_rate_phi', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_rho_prime', 'decay_rate_rho_prime',
        ])

        initial_parameters = perturb_model_parameters(
            initial_parameters,
            perturbation_size=0.8, perturbation_size_resonances=0.5,
            use_handpicked_bounds=True,
        )
        pipeline = make_pipeline_restricted(
            charged_ts, charged_cross_sections_values, charged_errors,
            neutral_ts, neutral_cross_sections_values, neutral_errors,
            kaon_mass, alpha, hc_squared,
            initial_parameters, path_to_reports, name=name, handpicked=True)
        return pipeline.run()

    final_results = []
    with Pool(processes=8) as pool:
        results = [pool.apply_async(f, (f'pp1_{i}',)) for i in range(10)]
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
