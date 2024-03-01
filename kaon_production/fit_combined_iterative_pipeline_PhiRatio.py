from configparser import ConfigParser
import gc

from kaon_production.data import (
    read_data_files_new, merge_statistical_and_systematic_errors, make_function_to_remove_fsr_effects)
from model_parameters import KaonParametersPhiRatio, Parameter
from pipeline.KaonCombinedIterativePipeline import KaonCombinedIterativePipeline
from common.utils import perturb_model_parameters


def make_initial_parameters(t_0_isoscalar, t_0_isovector):
    return KaonParametersPhiRatio(
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=1.0,
        t_in_isovector=1.0,
        a_omega=0.0,
        mass_omega=0.78266,
        decay_rate_omega=0.00868,
        a_omega_double_prime=0.025,
        mass_omega_double_prime=1.670,
        decay_rate_omega_double_prime=0.315,
        a_phi_charged=0.4,
        a_phi_neutral=0.4,
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
        mass_rho_double_prime=1.719,  # 1.6888,
        decay_rate_rho_double_prime=0.490,  # 0.161
    )
    # return KaonParametersFixedSelected.from_list([
    #     # Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
    #     # Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
    #     # Parameter(name='t_in_isoscalar', value=3.14, is_fixed=False),
    #     # Parameter(name='t_in_isovector', value=4.51, is_fixed=False),
    #     # Parameter(name='a_omega', value=0.281597535832303, is_fixed=False),
    #     # Parameter(name='mass_omega', value=0.78266, is_fixed=True),
    #     # Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
    #     # Parameter(name='a_omega_prime', value=-0.1078676216790596, is_fixed=False),
    #     # Parameter(name='mass_omega_prime', value=1.42, is_fixed=True),
    #     # Parameter(name='decay_rate_omega_prime', value=0.15, is_fixed=False),
    #     # Parameter(name='a_omega_double_prime', value=0.1385870881083241, is_fixed=False),
    #     # Parameter(name='mass_omega_double_prime', value=1.67, is_fixed=True),
    #     # Parameter(name='decay_rate_omega_double_prime', value=0.35, is_fixed=False),
    #     # Parameter(name='a_phi', value=0.2919572632095028, is_fixed=False),
    #     # Parameter(name='mass_phi', value=1.0190898268719202, is_fixed=False),
    #     # Parameter(name='decay_rate_phi', value=0.0043075044510112095, is_fixed=False),
    #     # Parameter(name='a_phi_prime', value=-0.1063075917575529, is_fixed=False),
    #     # Parameter(name='mass_phi_prime', value=1.6779405787843227, is_fixed=False),
    #     # Parameter(name='decay_rate_phi_prime', value=0.15, is_fixed=False),
    #     # Parameter(name='mass_phi_double_prime', value=2.1811221329851627, is_fixed=False),
    #     # Parameter(name='decay_rate_phi_double_prime', value=0.07602092049412262, is_fixed=False),
    #     # Parameter(name='a_rho', value=0.4085860041596347, is_fixed=False),
    #     # Parameter(name='mass_rho', value=0.75823, is_fixed=True),
    #     # Parameter(name='decay_rate_rho', value=0.14456, is_fixed=True),
    #     # Parameter(name='a_rho_prime', value=0.2278416614723415, is_fixed=False),
    #     # Parameter(name='mass_rho_prime', value=1.25, is_fixed=False),
    #     # Parameter(name='decay_rate_rho_prime', value=0.4295734885140618, is_fixed=False),
    #     # Parameter(name='a_rho_double_prime', value=-0.02922483846581725, is_fixed=False),
    #     # Parameter(name='mass_rho_double_prime', value=1.6826881404960172, is_fixed=False),
    #     # Parameter(name='decay_rate_rho_double_prime', value=0.1536976816912829, is_fixed=False),
    #     # Parameter(name='mass_rho_triple_prime', value=2.1235043993928477, is_fixed=False),
    #     # Parameter(name='decay_rate_rho_triple_prime', value=0.5, is_fixed=False),
    #     Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
    #     Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
    #     Parameter(name='t_in_isoscalar', value=4.5, is_fixed=False),
    #     Parameter(name='t_in_isovector', value=4.5, is_fixed=False),
    #     Parameter(name='a_omega', value=0.273221501337684, is_fixed=False),
    #     Parameter(name='mass_omega', value=0.78266, is_fixed=True),
    #     Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
    #     Parameter(name='a_omega_prime', value=-0.09837038062184843, is_fixed=False),
    #     Parameter(name='mass_omega_prime', value=1.42, is_fixed=True),
    #     Parameter(name='decay_rate_omega_prime', value=0.29, is_fixed=False),
    #     Parameter(name='a_omega_double_prime', value=0.06437550469737774, is_fixed=False),
    #     Parameter(name='mass_omega_double_prime', value=1.67, is_fixed=True),
    #     Parameter(name='decay_rate_omega_double_prime', value=0.4891490612315445, is_fixed=False),
    #     Parameter(name='a_phi', value=0.2920397094901536, is_fixed=False),
    #     Parameter(name='mass_phi', value=1.0190930163018095, is_fixed=False),
    #     Parameter(name='decay_rate_phi', value=0.0043119670480319, is_fixed=False),
    #     Parameter(name='a_phi_prime', value=-0.032699127581405606, is_fixed=False),
    #     Parameter(name='mass_phi_prime', value=1.6485253140438345, is_fixed=False),
    #     Parameter(name='decay_rate_phi_prime', value=0.1866983890167749, is_fixed=False),
    #     Parameter(name='mass_phi_double_prime', value=2.1998888333530577, is_fixed=False),
    #     Parameter(name='decay_rate_phi_double_prime', value=0.07, is_fixed=False),
    #     Parameter(name='a_rho', value=0.42029668396413666, is_fixed=False),
    #     Parameter(name='mass_rho', value=0.75823, is_fixed=True),
    #     Parameter(name='decay_rate_rho', value=0.14456, is_fixed=True),
    #     Parameter(name='a_rho_prime', value=0.22034081751234255, is_fixed=False),
    #     Parameter(name='mass_rho_prime', value=1.4, is_fixed=False),
    #     Parameter(name='decay_rate_rho_prime', value=0.43596469243817315, is_fixed=False),
    #     Parameter(name='a_rho_double_prime', value=-0.02944848197348991, is_fixed=False),
    #     Parameter(name='mass_rho_double_prime', value=1.6911378888406943, is_fixed=False),
    #     Parameter(name='decay_rate_rho_double_prime', value=0.15008632300424707, is_fixed=False),
    #     Parameter(name='mass_rho_triple_prime', value=2.113696787994467, is_fixed=False),
    #     Parameter(name='decay_rate_rho_triple_prime', value=0.5, is_fixed=False),
    # ])


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports/kaons'

    remove_fsr_effects = make_function_to_remove_fsr_effects(charged_kaon_mass, alpha)

    def discard_above_threshold(threshold, xs, ys, ers):
        return list(zip(*filter(lambda t: t[0] < threshold, zip(xs, ys, ers))))

    THRESHOLD = 10  # GeV^2

    (timelike_charged_ts, timelike_charged_cross_sections_values,
     timelike_charged_errors) = discard_above_threshold(THRESHOLD, *remove_fsr_effects(
        *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    # 'babar_2013_charged_kaons_undressed.csv',
                    # 'cmd_3_charged_kaons_undressed.csv',
                    # 'snd_charged_kaons_undressed.csv',
                    # 'babar_charged_kaons_2015_undressed.csv',
                    'cmd_3_charged_kaons_undressed.csv',
                    'babar_2013_charged_kaons_undressed.csv',
                    'BESIII_charged_kaons_2019_undressed.csv',
                ]
            )
        )
    ))
    (timelike_neutral_ts, timelike_neutral_cross_sections_values,
     timelike_neutral_errors) = discard_above_threshold(THRESHOLD, *merge_statistical_and_systematic_errors(
            *read_data_files_new(
                file_names=[
                    # 'cmd_2_neutral_kaons_undressed.csv',
                    # 'cmd_3_neutral_kaons_undressed.csv',
                    # 'snd_neutral_kaons_charged_mode_undressed.csv',
                    # 'snd_neutral_kaons_neutral_mode_undressed.csv',
                    'cmd_3_neutral_kaons_undressed.csv',
                    'babar_neutral_kaons_2014_undressed.csv',
                    'BESIII_neutral_kaons_2021_undressed.csv',
                ]
            )
    ))

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
            #  perturbation_size=0.2, perturbation_size_resonances=0.1,
            perturbation_size=1.0, perturbation_size_resonances=0.1,
            use_handpicked_bounds=True,
        )
        # numbers = (5, 3, 5, 2, 7, 5, 10, 15)
        # repetitions = (10, 40, 20, 15, 30, 25, 30, 20)
        numbers = (5, 3, 4, 10, 5, 2, 10, 6, 12)
        repetitions = (10, 40, 20, 30, 20, 10, 20, 30, 20)
        pipeline = KaonCombinedIterativePipeline(
            name, initial_parameters,
            charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, path_to_reports,
            t_cs_values_charged=timelike_charged_ts,
            cross_sections_charged=timelike_charged_cross_sections_values,
            cs_errors_charged=timelike_charged_errors,
            t_cs_values_neutral=timelike_neutral_ts,
            cross_sections_neutral=timelike_neutral_cross_sections_values,
            cs_errors_neutral=timelike_neutral_errors,
            t_ff_values_charged=spacelike_charged_ts,
            form_factors_charged=spacelike_charged_form_factor_values,
            ff_errors_charged=space_charged_errors,
            plot=False, use_handpicked_bounds=True,
            nr_free_params=numbers, nr_iterations=repetitions,
            nr_initial_rounds_with_fixed_resonances=10,
            nr_initial_rounds_on_partial_dataset=120,
            fit_on_timelike_data_only=True,
        )
        return pipeline.run()

    final_results = []
    best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
    for i in range(0, 5):
        result = f(f'runPhiRatio1_{i}')
        gc.collect()
        print(result)

        if result and result.get('chi_squared', None) is not None:
            final_results.append(result)
            if best_fit['chi_squared'] is None or float(result['chi_squared']) < float(best_fit['chi_squared']):
                best_fit = result
    print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: float(fr['chi_squared']))[:3]:
        print(final_result)
