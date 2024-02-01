from configparser import ConfigParser

from kaon_production.data import (
    read_data_files_new, merge_statistical_and_systematic_errors, make_function_to_apply_fsr_corrections)
from model_parameters import KaonParametersFixedSelected
from pipeline.KaonCombinedIterativePipeline import KaonCombinedIterativePipeline
from common.utils import perturb_model_parameters


def make_initial_parameters(t_0_isoscalar, t_0_isovector):
    return KaonParametersFixedSelected(
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=1.0,
        t_in_isovector=1.0,
        a_omega=0.0,
        mass_omega=0.78266,
        decay_rate_omega=0.00868,
        a_omega_prime=0.025,
        mass_omega_prime=1.420,
        decay_rate_omega_prime=0.29,
        a_omega_double_prime=0.025,
        mass_omega_double_prime=1.670,
        decay_rate_omega_double_prime=0.315,
        a_phi=0.4,
        mass_phi=1.019461,
        decay_rate_phi=0.004249,
        a_phi_prime=0.025,
        mass_phi_prime=1.680,
        decay_rate_phi_prime=0.150,
        mass_phi_double_prime=2.159,
        decay_rate_phi_double_prime=0.137,
        a_rho=0.0,
        mass_rho=0.75823,  # 0.77526,
        decay_rate_rho=0.14456,  # 0.1474,
        a_rho_prime=1.0/6,
        mass_rho_prime=1.342,  # 1.465,
        decay_rate_rho_prime=0.492,  # 0.4,
        a_rho_double_prime=1.0/6,
        mass_rho_double_prime=1.719,  # 1.6888,
        decay_rate_rho_double_prime=0.490,  # 0.161,
        mass_rho_triple_prime=2.15,
        decay_rate_rho_triple_prime=0.3,
    )


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports/kaons'

    apply_fsr_correction = make_function_to_apply_fsr_corrections(kaon_mass, alpha)

    (timelike_charged_ts, timelike_charged_cross_sections_values,
     timelike_charged_errors) = apply_fsr_correction(
        *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    'babar_2013_charged_kaons_undressed.csv',
                    'cmd_3_charged_kaons_undressed.csv',
                    'snd_charged_kaons_undressed.csv',
                ]
            )
        )
    )
    (timelike_neutral_ts, timelike_neutral_cross_sections_values,
     timelike_neutral_errors) = apply_fsr_correction(
        *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    'cmd_3_neutral_kaons_undressed.csv',
                    'snd_neutral_kaons_charged_mode_undressed.csv',
                    'snd_neutral_kaons_neutral_mode_undressed.csv',
                ]
            )
        )
    )
    (spacelike_charged_ts, spacelike_charged_form_factor_values,
     space_charged_errors) = merge_statistical_and_systematic_errors(
        *read_data_files_new(
            file_names=[
                'spacelike_charged_kaons_formfactor_1980_undressed.csv',
                'spacelike_charged_kaons_formfactor_1986_undressed.csv',
            ]
        )
    )
    print(f'Length of timelike charged kaons data: {len(timelike_charged_ts)}')
    print(f'Length of timelike neutral kaons data: {len(timelike_neutral_ts)}')
    print(f'Length of spacelike charged kaons data: {len(spacelike_charged_ts)}')


    def f(name):
        initial_parameters = make_initial_parameters(t_0_isoscalar, t_0_isovector)

        initial_parameters = perturb_model_parameters(
            initial_parameters,
            perturbation_size=1.0, perturbation_size_resonances=0.1,
            use_handpicked_bounds=False,
        )
        numbers = (5, 3, 5, 2, 7, 5, 10, 15)
        repetitions = (10, 40, 20, 5, 25, 20, 30, 20)
        pipeline = KaonCombinedIterativePipeline(
            name, initial_parameters,
            kaon_mass, alpha, hc_squared, path_to_reports,
            t_cs_values_charged=timelike_charged_ts,
            cross_sections_charged=timelike_charged_cross_sections_values,
            cs_errors_charged=timelike_charged_errors,
            t_cs_values_neutral=timelike_neutral_ts,
            cross_sections_neutral=timelike_neutral_cross_sections_values,
            cs_errors_neutral=timelike_neutral_errors,
            t_ff_values_charged=spacelike_charged_ts,
            form_factors_charged=spacelike_charged_form_factor_values,
            ff_errors_charged=space_charged_errors,
            plot=False, use_handpicked_bounds=False,
            nr_free_params=numbers, nr_iterations=repetitions,
            nr_initial_rounds_with_fixed_resonances=10,
            fit_on_timelike_data_only=True,
        )
        return pipeline.run()

    final_results = []
    best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
    for i in range(100):
        result = f(f'run1_{i}')
        print(result)

        if result and result.get('chi_squared', None) is not None:
            final_results.append(result)
            if best_fit['chi_squared'] is None or result['chi_squared'] < best_fit['chi_squared']:
                best_fit = result
    print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: fr['chi_squared'])[:3]:
        print(final_result)
