import os
import os.path
import statistics
from configparser import ConfigParser
import random

from kaon_production.data import (
    read_data_files_new, merge_statistical_and_systematic_errors,
    make_function_to_remove_fsr_effects, generate_monte_carlo_data_sample,
)
from model_parameters import (KaonParametersFixedSelected, KaonParametersPhiRatio, KaonParametersSimplified,
                              Parameter, PionParameters)
from pipeline.KaonCombinedIterativePipeline import KaonCombinedIterativePipeline


def _merge(list_of_datafiles):
    xs, ys, stat_errs, sys_errs = [], [], [], []
    for datafile in list_of_datafiles:
        xs.extend(datafile[0])
        ys.extend(datafile[1])
        stat_errs.extend(datafile[2])
        sys_errs.extend(datafile[3])
    return xs, ys, stat_errs, sys_errs


def _generate_data_set(
        files_charged_timelike, files_neutral_timelike, files_charged_spacelike,
        remove_fsr_effects_function
):
    timelike_charged_data = _merge([
        generate_monte_carlo_data_sample(*read_data_files_new([filepath]))
        for filepath in files_charged_timelike
    ])
    (timelike_charged_ts, timelike_charged_cross_sections_values,
     timelike_charged_errors) = remove_fsr_effects_function(
        *merge_statistical_and_systematic_errors(*timelike_charged_data)
    )

    timelike_neutral_data = _merge([
        generate_monte_carlo_data_sample(*read_data_files_new([filepath]))
        for filepath in files_neutral_timelike
    ])
    (timelike_neutral_ts, timelike_neutral_cross_sections_values,
     timelike_neutral_errors) = merge_statistical_and_systematic_errors(*timelike_neutral_data)

    (spacelike_charged_ts, spacelike_charged_form_factor_values,
     space_charged_errors) = merge_statistical_and_systematic_errors(
        *read_data_files_new(files_charged_spacelike)
    )
    return (
        timelike_charged_ts, timelike_charged_cross_sections_values,
        timelike_charged_errors, timelike_neutral_ts, timelike_neutral_cross_sections_values,
        timelike_neutral_errors, spacelike_charged_ts, spacelike_charged_form_factor_values,
        space_charged_errors,
    )


def _run_pipeline(save_dir, name, starting_parameters,
                  charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared,
                  timelike_charged_ts,
                  timelike_charged_cross_sections_values, timelike_charged_errors,
                  timelike_neutral_ts, timelike_neutral_cross_sections_values,
                  timelike_neutral_errors, spacelike_charged_ts, spacelike_charged_form_factor_values,
                  spacelike_charged_errors):
    numbers = (7, 15, 10, 15)
    repetitions = (5, 2, 3, 4)
    pipeline = KaonCombinedIterativePipeline(
        name, starting_parameters,
        charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, save_dir,
        t_cs_values_charged=timelike_charged_ts,
        cross_sections_charged=timelike_charged_cross_sections_values,
        cs_errors_charged=timelike_charged_errors,
        t_cs_values_neutral=timelike_neutral_ts,
        cross_sections_neutral=timelike_neutral_cross_sections_values,
        cs_errors_neutral=timelike_neutral_errors,
        t_ff_values_charged=spacelike_charged_ts,
        form_factors_charged=spacelike_charged_form_factor_values,
        ff_errors_charged=spacelike_charged_errors,
        plot=False, use_handpicked_bounds=False,
        nr_free_params=numbers, nr_iterations=repetitions,
        nr_initial_rounds_with_fixed_resonances=2,
        nr_initial_rounds_on_partial_dataset=5,
        fit_on_timelike_data_only=True,
    )
    return pipeline.run()


def _generate_monte_carlo_parameters(
        original_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared,
        files_charged_timelike, files_neutral_timelike, files_charged_spacelike,
        remove_fsr_effects_function, nr_to_generate, save_dir, dir_exist_ok=False, start_n=0):
    os.makedirs(save_dir, exist_ok=dir_exist_ok)
    for n in range(start_n, start_n + nr_to_generate):
        name = f'item_{n}'
        _run_pipeline(
            save_dir, name, original_parameters.copy(),
            charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared,
            *_generate_data_set(files_charged_timelike, files_neutral_timelike,
                                files_charged_spacelike, remove_fsr_effects_function)
        )


def _generate_monte_carlo_parameters_from_errors(
        original_parameters, parameter_errors, nr_to_generate, save_dir, dir_exist_ok=False):
    for n in range(0, nr_to_generate):
        name = f'item_{n}'
        os.makedirs(os.path.join(save_dir, name), exist_ok=dir_exist_ok)
        res = []
        for par in original_parameters:
            if par.is_fixed:
                assert parameter_errors.get(par.name, 0) == 0
                res.append(Parameter(par.name, par.value, par.is_fixed))
            else:
                error = parameter_errors[par.name]
                new_value = random.gauss(mu=par.value, sigma=error)
                res.append(Parameter(par.name, new_value, par.is_fixed))
        new_parameters = type(original_parameters).from_list(res)
        print(f'Generated parameters {name}: {new_parameters.to_list()}')
        new_parameters.serialize_parameters_into(
            os.path.join(save_dir, name, 'final_fit_parameters.pickle')
        )


def _read_parameters_in_dir(dirpath, pion=False):
    filenames = os.listdir(dirpath)
    if pion:
        parameters_list = [
            PionParameters.load_from_serialized_parameters(
                os.path.join(dirpath, filename, 'final_fit_parameters.pickle')
            ) for filename in filenames
        ]
    else:
        parameters_list = [
            KaonParametersSimplified.load_from_serialized_parameters(
                os.path.join(dirpath, filename, 'final_fit_parameters.pickle')
            ) for filename in filenames
        ]
    print(f'Loaded {len(parameters_list)} sets of parameters from {dirpath}')
    return parameters_list


def _calculate_parameter_mean_and_std(list_pars):
    if not list_pars:
        return None
    names = [parameter.name for parameter in list_pars[0]]
    result = {}
    for name in names:
        vals = [pars[name].value for pars in list_pars]
        result[name] = {'mean': statistics.mean(vals), 'standard_deviation': statistics.stdev(vals)}
    return result


def _calculate_mean_and_std_of_function_values(f, list_pars):
    values = [f(parameters) for parameters in list_pars]
    return {'mean': statistics.mean(values), 'standard_deviation': statistics.stdev(values)}


def report(msg: str, filepath: str):
    with open(filepath, 'a') as f:
        f.write(msg + '\n')
    print(msg)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    charged_pion_mass = config.getfloat('constants', 'charged_pion_mass')
    muon_mass = config.getfloat('constants', 'muon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    remove_fsr_effects = make_function_to_remove_fsr_effects(charged_kaon_mass, alpha)

    files_charged_timelike = [
        'cmd_3_charged_kaons_undressed.csv',  # added
        'babar_2013_charged_kaons_undressed.csv',
        'babar_charged_kaons_2015_undressed.csv',
        'BESIII_charged_kaons_2019_undressed.csv',
    ]
    files_neutral_timelike = [
        'cmd_2_neutral_kaons_undressed.csv',  # added
        'cmd_3_neutral_kaons_undressed.csv',
        'babar_neutral_kaons_2014_undressed.csv',
        'BESIII_neutral_kaons_2021_undressed.csv',
    ]
    files_charged_spacelike = [
        #'spacelike_charged_kaons_formfactor_1980_undressed.csv',
        #'spacelike_charged_kaons_formfactor_1986_undressed.csv',
    ]

    source_pars_directory = '/home/lukas/reports/kaons/article2_fit'
    # save_dir = '/home/lukas/reports/kaons/article2_fit/pion_parameters/monte_carlo'
    # save_dir = os.path.join(source_pars_directory, 'monte_carlo')
    original_parameters = KaonParametersSimplified.load_from_serialized_parameters(
       os.path.join(source_pars_directory, 'final_fit_parameters.pickle')
    )
    original_parameters_pion = PionParameters.from_list([
        Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
        Parameter(name='t_in_isovector', value=1.2733, is_fixed=False),
        Parameter(name='mass_rho', value=0.7621, is_fixed=False),
        Parameter(name='decay_rate_rho', value=0.14423672, is_fixed=False),
        Parameter(name='a_rho_prime', value=-0.07060638, is_fixed=False),
        Parameter(name='mass_rho_prime', value=1.3500, is_fixed=False),
        Parameter(name='decay_rate_rho_prime', value=0.3319913, is_fixed=False),
        Parameter(name='a_rho_double_prime', value=0.05785514, is_fixed=False),
        Parameter(name='mass_rho_double_prime', value=1.76928672, is_fixed=False),
        Parameter(name='decay_rate_rho_double_prime', value=0.25311443, is_fixed=False),
        Parameter(name='a_rho_triple_prime', value=0.00208887, is_fixed=False),
        Parameter(name='mass_rho_triple_prime', value=2.24674832, is_fixed=False),
        Parameter(name='decay_rate_rho_triple_prime', value=0.0700, is_fixed=False),
        Parameter(name='w_pole', value=0.38329263, is_fixed=False),
        Parameter(name='w_zero', value=0.2844582, is_fixed=False),
    ])
    #
    # parameter_errors = {
    #     't_0_isovector': 0.0,
    #     't_in_isovector': 0.013,
    #     'mass_rho': 0.0080,
    #     'decay_rate_rho': 0.0014,
    #     'a_rho_prime': 0.0012,
    #     'mass_rho_prime': 0.011,
    #     'decay_rate_rho_prime': 0.0033,
    #     'a_rho_double_prime': 0.0010,
    #     'mass_rho_double_prime': 0.018,
    #     'decay_rate_rho_double_prime': 0.0025,
    #     'a_rho_triple_prime': 0.0005,
    #     'mass_rho_triple_prime': 0.011,
    #     'decay_rate_rho_triple_prime': 0.0007,
    #     'w_pole': 0.0060,
    #     'w_zero': 0.0033,
    # }
    #
    # _generate_monte_carlo_parameters_from_errors(
    #     original_parameters, parameter_errors, 1000, save_dir, True
    # )

    #
    # _generate_monte_carlo_parameters(
    #      original_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared,
    #      files_charged_timelike, files_neutral_timelike, files_charged_spacelike,
    #      remove_fsr_effects, 10, save_dir, dir_exist_ok=True, start_n=405,
    # )

    # report_filepath = '/home/lukas/git_repos/UA_model/charge_radii_report.txt'
    # report('Parameters statistics:', report_filepath)
    # report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir))), report_filepath)
    #
    # report('Charge radii:', report_filepath)
    #
    # from calculate_charge_radius import wrap_partial_form_factor_function, calculate_charge_radius
    # from common.utils import make_partial_form_factor_for_parameters
    #
    # def make_calculate_charge_radius_from_parameters(charged):
    #     def f(parameters):
    #         parameters.fix_all_parameters()
    #         partial = make_partial_form_factor_for_parameters(parameters, return_absolute_value=False)
    #         ff = wrap_partial_form_factor_function(partial, charged=charged)
    #         return calculate_charge_radius(ff, hc_squared)
    #     return f
    #
    #
    # charged_f = make_calculate_charge_radius_from_parameters(charged=True)
    # charge_radii_statistics = _calculate_mean_and_std_of_function_values(
    #     charged_f, _read_parameters_in_dir(save_dir),
    # )
    # report('Charged', report_filepath)
    # report(str(charge_radii_statistics), report_filepath)
    # report('Fit: ' + str(charged_f(original_parameters)), report_filepath)
    #
    # neutral_f = make_calculate_charge_radius_from_parameters(charged=False)
    # charge_radii_statistics = _calculate_mean_and_std_of_function_values(
    #     neutral_f, _read_parameters_in_dir(save_dir),
    # )
    # report('Neutral', report_filepath)
    # report(str(charge_radii_statistics), report_filepath)
    # report('Fit: ' + str(neutral_f(original_parameters)), report_filepath)
    #

    #
    # from calculate_r_ratio import calculate_cross_sections_ratio_at_phi_peak, calculate_r_ratio
    #
    # def get_r_ratio(pars):
    #     return calculate_r_ratio(pars, charged_kaon_mass, neutral_kaon_mass, alpha, True)
    #
    #
    # def get_r_ratio_no_rc(pars):
    #     return calculate_r_ratio(pars, charged_kaon_mass, neutral_kaon_mass, alpha, False)
    #
    # def get_cs_ratio(pars):
    #     return calculate_cross_sections_ratio_at_phi_peak(
    #         pars, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, False,
    #     )
    #
    # print(f'r_ratio: {_calculate_mean_and_std_of_function_values(get_r_ratio,  _read_parameters_in_dir(save_dir))}')
    # print(f'r_ratio_no_rc: {_calculate_mean_and_std_of_function_values(get_r_ratio_no_rc,  _read_parameters_in_dir(save_dir))}')
    # print(f'cs_ratio: {_calculate_mean_and_std_of_function_values(get_cs_ratio, _read_parameters_in_dir(save_dir))}')

    # save_dir_pion = '/home/lukas/reports/kaons/article2_fit/pion_parameters/monte_carlo'
    # save_dir_kaon = '/home/lukas/reports/kaons/article2_fit/monte_carlo'
    # report_filepath = '/home/lukas/git_repos/UA_model/em_mass_report.txt'
    # report('Pion parameters statistics:', report_filepath)
    # report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir_pion, pion=True))), report_filepath)
    # report('Kaon parameters statistics:', report_filepath)
    # report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir_kaon))), report_filepath)
    #
    # from calculate_em_mass_contribution import (_wrap_partial_form_factor_function, _make_partial_for_pion_parameters,
    #                                             calculate_em_mass2_contribution)
    # from common.utils import make_partial_form_factor_for_parameters
    #
    # def make_calculate_em_mass2_from_parameters(for_kaon=True, charged=True, drop_int_error=True):
    #     def f(parameters):
    #         if for_kaon:
    #             parameters.fix_all_parameters()
    #             f = _wrap_partial_form_factor_function(
    #                 make_partial_form_factor_for_parameters(parameters, return_absolute_value=False),
    #                 charged=charged
    #             )
    #             mass = charged_kaon_mass if charged else neutral_kaon_mass
    #         else:
    #             f = _make_partial_for_pion_parameters(parameters)
    #             mass = charged_pion_mass
    #         res_int, err_int = calculate_em_mass2_contribution(f, alpha, mass)
    #         if drop_int_error:
    #             return res_int
    #         return res_int, err_int
    #     return f
    #
    # report('EM mass charged kaon:', report_filepath)
    # charged_kaon_f = make_calculate_em_mass2_from_parameters(for_kaon=True, charged=True)
    # charged_kaon_m2_statistics = _calculate_mean_and_std_of_function_values(
    #     charged_kaon_f, _read_parameters_in_dir(save_dir_kaon),
    # )
    # report('Charged kaon', report_filepath)
    # report(str(charged_kaon_m2_statistics), report_filepath)
    # charged_kaon_f2 = make_calculate_em_mass2_from_parameters(for_kaon=True, charged=True, drop_int_error=False)
    # report('Fit: ' + str(charged_kaon_f2(original_parameters)), report_filepath)
    #
    # report('EM mass neutral kaon:', report_filepath)
    # neutral_kaon_f = make_calculate_em_mass2_from_parameters(for_kaon=True, charged=False)
    # neutral_kaon_m2_statistics = _calculate_mean_and_std_of_function_values(
    #     neutral_kaon_f, _read_parameters_in_dir(save_dir_kaon),
    # )
    # report('Neutral kaon', report_filepath)
    # report(str(neutral_kaon_m2_statistics), report_filepath)
    # neutral_kaon_f2 = make_calculate_em_mass2_from_parameters(for_kaon=True, charged=False, drop_int_error=False)
    # report('Fit: ' + str(neutral_kaon_f2(original_parameters)), report_filepath)
    #
    # report('EM mass charged pion:', report_filepath)
    # charged_pion_f = make_calculate_em_mass2_from_parameters(for_kaon=False, charged=True)
    # charged_pion_m2_statistics = _calculate_mean_and_std_of_function_values(
    #     charged_pion_f, _read_parameters_in_dir(save_dir_pion, pion=True),
    # )
    # report('Charged pion', report_filepath)
    # report(str(charged_pion_m2_statistics), report_filepath)
    # charged_pion_f2 = make_calculate_em_mass2_from_parameters(for_kaon=False, charged=True, drop_int_error=False)
    # report('Fit: ' + str(charged_pion_f2(original_parameters_pion)), report_filepath)

    save_dir_pion = '/home/lukas/reports/kaons/article2_fit/pion_parameters/monte_carlo'
    save_dir_kaon = '/home/lukas/reports/kaons/article2_fit/monte_carlo'
    report_filepath = '/home/lukas/git_repos/UA_model/hvp_report.txt'
    report('Pion parameters statistics:', report_filepath)
    report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir_pion, pion=True))), report_filepath)
    report('Kaon parameters statistics:', report_filepath)
    report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir_kaon))), report_filepath)

    from calculate_hvp_contribution import make_calculate_hvp_contribution_kaon, make_calculate_hvp_contribution_pion

    report('HVP contribution charged kaon:', report_filepath)
    charged_kaon_f = make_calculate_hvp_contribution_kaon(
        alpha, hc_squared, muon_mass, charged_kaon_mass, neutral_kaon_mass, charged=True)
    charged_kaon_hvp_statistics = _calculate_mean_and_std_of_function_values(
        charged_kaon_f, _read_parameters_in_dir(save_dir_kaon),
    )
    report('Charged kaon', report_filepath)
    report(str(charged_kaon_hvp_statistics), report_filepath)
    report('Fit: ' + str(charged_kaon_f(original_parameters)), report_filepath)

    report('HVP contribution neutral kaon:', report_filepath)
    neutral_kaon_f = make_calculate_hvp_contribution_kaon(
        alpha, hc_squared, muon_mass, charged_kaon_mass, neutral_kaon_mass, charged=False)
    neutral_kaon_hvp_statistics = _calculate_mean_and_std_of_function_values(
        neutral_kaon_f, _read_parameters_in_dir(save_dir_kaon),
    )
    report('Neutral kaon', report_filepath)
    report(str(neutral_kaon_hvp_statistics), report_filepath)
    report('Fit: ' + str(neutral_kaon_f(original_parameters)), report_filepath)

    report('HVP contribution charged pion:', report_filepath)
    charged_pion_f = make_calculate_hvp_contribution_pion(
        alpha, hc_squared, muon_mass, charged_pion_mass, upper_limit=1.0,
    )
    charged_pion_hvp_statistics = _calculate_mean_and_std_of_function_values(
        charged_pion_f, _read_parameters_in_dir(save_dir_pion, pion=True),
    )
    report('Charged pion E<1.0GeV', report_filepath)
    report(str(charged_pion_hvp_statistics), report_filepath)
    report('Fit: ' + str(charged_pion_f(original_parameters_pion)), report_filepath)
