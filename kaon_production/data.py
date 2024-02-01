import csv
import math
import os.path
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable

from kaon_production.eta_correction import add_fsr_effects, remove_fsr_effects

KaonDatapoint = namedtuple('KaonDatapoint', 't is_charged is_for_cross_section')

DIR_NAME = '../data'

# TODO: remove the old read_data function?


def read_data(file_name: str = 'charged_kaon.csv') -> Tuple[List[float], List[float], List[float]]:
    xs = []
    ys = []
    errs = []

    filepath = os.path.join(DIR_NAME, file_name)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for x, y, err in reader:
            xs.append(float(x))
            ys.append(float(y))
            errs.append(float(err))

    return xs, ys, errs


def read_data_new(
        file_name: str, subdir_name: Optional[str] = 'new'
) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs = []
    ys = []
    stat_errs = []
    sys_errs = []

    filepath = os.path.join(DIR_NAME, subdir_name, file_name)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for x, y, stat_err, sys_err in reader:
            xs.append(float(x))
            ys.append(float(y))
            stat_errs.append(float(stat_err))
            sys_errs.append(float(sys_err))

    return xs, ys, stat_errs, sys_errs


def read_data_files_new(
        file_names: List[str], subdir_name: Optional[str] = 'new'
) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs = []
    ys = []
    stat_errs = []
    sys_errs = []

    for file_name in file_names:
        new_xs, new_ys, new_stat_errs, new_sys_errs = read_data_new(file_name, subdir_name)
        xs.extend(new_xs)
        ys.extend(new_ys)
        stat_errs.extend(new_stat_errs)
        sys_errs.extend(new_sys_errs)

    return xs, ys, stat_errs, sys_errs


def transform_energy_to_s(
        source_filepath: str,
        output_filepath: str,
        convert_from_pikobarns_to_nanobarns: Optional[bool] = False,
) -> None:
    converted = []

    with open(source_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for energy, cross_section, statistical_error, systematic_error in reader:
            if convert_from_pikobarns_to_nanobarns:
                cross_section = str(round(float(cross_section) / 1000, 6))
                statistical_error = str(round(float(statistical_error) / 1000, 6))
                systematic_error = str(round(float(systematic_error) / 1000, 6))
            converted.append(
                (str(round(float(energy)**2, 6)),
                 cross_section,
                 statistical_error,
                 systematic_error)
            )

    with open(output_filepath, 'w') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(converted)


def process_spacelike_data(source_filepath: str, output_filepath: str) -> None:
    """
    This function is to process spacelike data with columns of signature:
        /pm |q^2| [GeV^2], |FF|^2 [1], error [1]
    We need to multiply q^2 by -1 (in our convention space-like four-momenta
    have negative norm), calculate the square root of FF^2, and adjust the error
    appropriately.

    """
    def _square_root_error(original_error, original_value, round_to_digits):
        new_error = 0.5 * original_error / math.sqrt(original_value)
        return round(new_error, round_to_digits)

    converted = []

    round_to_digits = 5
    with open(source_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for abs_q2, ff2, stat_error, sys_error in reader:
            converted.append(
                (-1 * abs(float(abs_q2)),
                round(math.sqrt(float(ff2)), round_to_digits),
                _square_root_error(float(stat_error), float(ff2), round_to_digits),
                _square_root_error(float(sys_error), float(ff2), round_to_digits))
            )

    with open(output_filepath, 'w') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(converted)


def merge_statistical_and_systematic_errors(
        xs: List[float], ys: List[float], stat_errs: List[float], sys_errs: List[float]
) -> Tuple[List[float], List[float], List[float]]:
    assert len(xs) == len(ys) == len(stat_errs) == len(sys_errs)
    merged_errs = []
    for stat_e, sys_e in zip(stat_errs, sys_errs):
        merged_errs.append(math.sqrt(stat_e**2 + sys_e**2))
    return xs, ys, merged_errs


def make_function_to_add_fsr_effects(
        final_particle_mass: float, alpha: float,
) -> Callable[[List[float], List[float], List[float]], Tuple[List[float], List[float], List[float]]]:

    def apply_fsr_correction(
        s_list: List[float], cs_list: List[float], errs: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        assert len(s_list) == len(cs_list) == len(errs)
        corrected_cs = []
        corrected_errs = []
        for s, cs, err in zip(s_list, cs_list, errs):
            corrected_cs.append(add_fsr_effects(cs, s, final_particle_mass, alpha))
            corrected_errs.append(add_fsr_effects(err, s, final_particle_mass, alpha))
        return s_list, corrected_cs, corrected_errs

    return apply_fsr_correction


def make_function_to_remove_fsr_effects(
        final_particle_mass: float, alpha: float,
) -> Callable[[List[float], List[float], List[float]], Tuple[List[float], List[float], List[float]]]:

    def apply_fsr_correction(
        s_list: List[float], cs_list: List[float], errs: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        assert len(s_list) == len(cs_list) == len(errs)
        corrected_cs = []
        corrected_errs = []
        for s, cs, err in zip(s_list, cs_list, errs):
            corrected_cs.append(remove_fsr_effects(cs, s, final_particle_mass, alpha))
            corrected_errs.append(remove_fsr_effects(err, s, final_particle_mass, alpha))
        return s_list, corrected_cs, corrected_errs

    return apply_fsr_correction


def plot_data(
        filenames: List[str],
        subdir_name: str,
        title: str,
        x_axis_label: str,
        y_axis_label: str
):

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    for filename in filenames:
        xs, ys, errs = merge_statistical_and_systematic_errors(*read_data_new(filename, subdir_name))
        ax.errorbar(xs, ys, yerr=errs, fmt='x', label=filename)

    ax.legend(loc='upper right')

    plt.show()
    plt.close()


if __name__ == '__main__':
    # file_name = 'babar_neutral_kaons_2014_undressed.csv'
    # transform_energy_to_s(
    #     f'../data/raw_files/{file_name}',
    #     f'../data/new/{file_name}',
    #     convert_from_pikobarns_to_nanobarns=False,
    # )

    filenames = [
        'cmd_3_charged_kaons_undressed.csv',
        'snd_charged_kaons_undressed.csv',
        'babar_2013_charged_kaons_undressed.csv',
        'babar_charged_kaons_2015_undressed.csv',
        'BESIII_charged_kaons_2019_undressed.csv',
    ]
    # filenames = [
    #     'cmd_3_neutral_kaons_undressed.csv',
    #     'cmd_2_neutral_kaons_undressed.csv',
    #     'snd_neutral_kaons_charged_mode_undressed.csv',
    #     'snd_neutral_kaons_neutral_mode_undressed.csv',
    #     'babar_neutral_kaons_2014_undressed.csv',
    #     'BESIII_neutral_kaons_2021_undressed.csv',
    # ]
    plot_data(filenames, 'new', 'Cross sections --- Neutral kaons', 's[GeV^2]', 'sigma[nb]')

    # process_spacelike_data(
    #     f'../data/raw_files/spacelike_charged_kaons_formfactor2_1986_undressed.csv',
    #     f'../data/new/spacelike_charged_kaons_formfactor_1986_undressed.csv')
