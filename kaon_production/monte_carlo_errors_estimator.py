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


def _read_parameters_in_dir(dirpath):
    filenames = os.listdir(dirpath)
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
    save_dir = os.path.join(source_pars_directory, 'monte_carlo')
    original_parameters = KaonParametersSimplified.load_from_serialized_parameters(
       os.path.join(source_pars_directory, 'final_fit_parameters.pickle')
    )
    #
    # _generate_monte_carlo_parameters(
    #      original_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared,
    #      files_charged_timelike, files_neutral_timelike, files_charged_spacelike,
    #      remove_fsr_effects, 10, save_dir, dir_exist_ok=True, start_n=405,
    # )

    report_filepath = '/home/lukas/git_repos/UA_model/charge_radii_report.txt'
    report('Parameters statistics:', report_filepath)
    report(str(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir))), report_filepath)

    report('Charge radii:', report_filepath)

    from calculate_charge_radius import wrap_partial_form_factor_function, calculate_charge_radius
    from common.utils import make_partial_form_factor_for_parameters

    def make_calculate_charge_radius_from_parameters(charged):
        def f(parameters):
            parameters.fix_all_parameters()
            partial = make_partial_form_factor_for_parameters(parameters, return_absolute_value=False)
            ff = wrap_partial_form_factor_function(partial, charged=charged)
            return calculate_charge_radius(ff, hc_squared)
        return f


    charged_f = make_calculate_charge_radius_from_parameters(charged=True)
    charge_radii_statistics = _calculate_mean_and_std_of_function_values(
        charged_f, _read_parameters_in_dir(save_dir),
    )
    report('Charged', report_filepath)
    report(str(charge_radii_statistics), report_filepath)
    report('Fit: ' + str(charged_f(original_parameters)), report_filepath)

    neutral_f = make_calculate_charge_radius_from_parameters(charged=False)
    charge_radii_statistics = _calculate_mean_and_std_of_function_values(
        neutral_f, _read_parameters_in_dir(save_dir),
    )
    report('Neutral', report_filepath)
    report(str(charge_radii_statistics), report_filepath)
    report('Fit: ' + str(neutral_f(original_parameters)), report_filepath)


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
