import os
import os.path
import statistics
from configparser import ConfigParser

from kaon_production.data import (
    read_data_files_new, merge_statistical_and_systematic_errors,
    make_function_to_remove_fsr_effects, generate_monte_carlo_data_sample,
)
from model_parameters import KaonParametersFixedSelected
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
     timelike_neutral_errors) = remove_fsr_effects_function(
        *merge_statistical_and_systematic_errors(*timelike_neutral_data)
    )

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
                  kaon_mass, alpha, hc_squared,
                  timelike_charged_ts,
                  timelike_charged_cross_sections_values, timelike_charged_errors,
                  timelike_neutral_ts, timelike_neutral_cross_sections_values,
                  timelike_neutral_errors, spacelike_charged_ts, spacelike_charged_form_factor_values,
                  spacelike_charged_errors):
    numbers = (7, 15, 10, 15)
    repetitions = (5, 2, 3, 4)
    pipeline = KaonCombinedIterativePipeline(
        name, starting_parameters,
        kaon_mass, alpha, hc_squared, save_dir,
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
        fit_on_timelike_data_only=True,
    )
    return pipeline.run()


def _generate_monte_carlo_parameters(
        original_parameters, kaon_mass, alpha, hc_squared, files_charged_timelike, files_neutral_timelike,
        files_charged_spacelike, remove_fsr_effects_function, nr_to_generate, save_dir):
    os.makedirs(save_dir, exist_ok=False)
    for n in range(nr_to_generate):
        name = f'item_{n}'
        _run_pipeline(
            save_dir, name, original_parameters,
            kaon_mass, alpha, hc_squared,
            *_generate_data_set(files_charged_timelike, files_neutral_timelike,
                                files_charged_spacelike, remove_fsr_effects_function)
        )


def _read_parameters_in_dir(dirpath):
    filenames = os.listdir(dirpath)
    return [
        KaonParametersFixedSelected.load_from_serialized_parameters(
            os.path.join(dirpath, filename, 'final_fit_parameters.pickle')
        ) for filename in filenames
    ]


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


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    remove_fsr_effects = make_function_to_remove_fsr_effects(kaon_mass, alpha)

    files_charged_timelike = [
        'babar_2013_charged_kaons_undressed.csv',
        'babar_charged_kaons_2015_undressed.csv',
        'BESIII_charged_kaons_2019_undressed.csv',
    ]
    files_neutral_timelike = [
        'cmd_3_neutral_kaons_undressed.csv',
        'babar_neutral_kaons_2014_undressed.csv',
        'BESIII_neutral_kaons_2021_undressed.csv',
    ]
    files_charged_spacelike = [
        'spacelike_charged_kaons_formfactor_1980_undressed.csv',
        'spacelike_charged_kaons_formfactor_1986_undressed.csv',
    ]

    source_pars_directory = '/home/lukas/reports/kaons/run4_3'
    save_dir = os.path.join(source_pars_directory, 'monte_carlo')
    original_parameters = KaonParametersFixedSelected.load_from_serialized_parameters(
        os.path.join(source_pars_directory, 'final_fit_parameters.pickle')
    )

    # _generate_monte_carlo_parameters(
    #     original_parameters, kaon_mass, alpha, hc_squared, files_charged_timelike, files_neutral_timelike,
    #     files_charged_spacelike, remove_fsr_effects, 5, save_dir
    # )

    print(_calculate_parameter_mean_and_std(_read_parameters_in_dir(save_dir)))
