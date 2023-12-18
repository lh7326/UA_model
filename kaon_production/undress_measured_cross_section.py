from typing import List, Tuple, Optional

from configparser import ConfigParser
import csv
import math
import matplotlib.pyplot as plt

# TODO: Move this to a shared module. Perhaps refactor. Add unit tests.


def error_for_addition_uncorrelated(err1: float, err2: float) -> float:
    return math.sqrt(err1**2 + err2**2)


def error_for_addition_correlated(err1: float, err2: float) -> float:
    return err1 + err2


def error_for_multiplication_uncorrelated(val1: float, err1: float, val2: float, err2: float) -> float:
    rel_err1 = err1 / val1
    rel_err2 = err2 / val2
    return val1 * val2 * error_for_addition_uncorrelated(rel_err1, rel_err2)


def error_for_multiplication_correlated(val1: float, err1: float, val2: float, err2: float) -> float:
    return val1 * err2 + val2 * err1


def _undress_cross_section_measurement(
        alpha_0: float,
        alpha_s_squared: float,
        alpha_squared_err: float,
        cs_val: float,
        cs_err_stat: Optional[float] = 0.0,
        cs_err_sys: Optional[float] = 0.0,
        verbose=True,
) -> Tuple[float, float, float]:
    factor = alpha_0**2 / alpha_s_squared
    if verbose:
        print(f'Factor (alpha(0)/alpha(s))^2: {factor}')
    factor_err = (factor / alpha_s_squared) * alpha_squared_err
    undressed_cs = factor * cs_val
    rescaled_stat_error = factor * cs_err_stat
    new_sys_err = error_for_multiplication_uncorrelated(factor, factor_err, cs_val, cs_err_sys)
    return undressed_cs, rescaled_stat_error, new_sys_err


def _read_data_running_alpha_real_parts() -> Tuple[List[float], List[float], List[float]]:
    filepath_real = '../data/running_alpha/real_parts.csv'

    energy_real = []
    alpha_real = []
    error_real = []
    with open(filepath_real, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for en_r, al_r, _, er_r, _ in reader:
            energy_real.append(float(en_r))
            alpha_real.append(float(al_r))
            error_real.append(float(er_r))
    return energy_real, alpha_real, error_real


def _read_data_running_alpha_imaginary_parts() -> Tuple[List[float], List[float], List[float]]:
    filepath_imaginary = '../data/running_alpha/imaginary_parts.csv'

    energy_imaginary = []
    alpha_imaginary = []
    error_imaginary = []
    with open(filepath_imaginary, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for en_i, al_i, al_i_low, al_i_high in reader:
            energy_imaginary.append(float(en_i))
            alpha_imaginary.append(float(al_i))
            error_imaginary.append((float(al_i_high) - float(al_i_low)) / 2.0)
    return energy_imaginary, alpha_imaginary, error_imaginary


def _get_running_alpha_data() -> List[Tuple[float, float, float]]:
    energy_real, alpha_real, error_real = _read_data_running_alpha_real_parts()
    energy_imaginary, alpha_imaginary, error_imaginary = _read_data_running_alpha_imaginary_parts()

    energy, alpha_squared, error = [], [], []
    for en_r, en_i, al_r, al_i, er_r, er_i in zip(energy_real, energy_imaginary, alpha_real,
                                                  alpha_imaginary, error_real, error_imaginary):
        assert en_r == en_i
        energy.append(en_r)
        alpha_squared.append(al_r**2 + al_i**2)
        error.append(error_for_addition_correlated(2 * al_r * er_r, 2 * al_i * er_i))

    return list(zip(energy, alpha_squared, error))


def undress_data_point(
        energy: float,
        cs: float,
        err_stat: float,
        err_sys: float,
        alpha_0: float,
        running_alpha_data: List[Tuple[float, float, float]]
) -> Tuple[float, float, float, float]:
    len_data = len(running_alpha_data)
    i = 0
    running_alpha_data_point = running_alpha_data[i]
    if running_alpha_data_point[0] > energy:
        raise 'Running alpha data start at higher energies'
    while running_alpha_data_point[0] < energy:
        i += 1
        if i == len_data:
            raise 'Running alpha data do not reach high enough energy'
        running_alpha_data_point = running_alpha_data[i]
    lower_energy, lower_val, lower_err = running_alpha_data[i - 1]
    higher_energy, higher_val, higher_err = running_alpha_data_point
    print(f'Energy={energy}. Lower data={lower_energy},{lower_val},{lower_err}.'
          f'Higher data energy={higher_energy},{higher_val},{higher_err}')
    interpol_par = (energy - lower_energy) / (higher_energy - lower_energy)
    aver_val = lower_val * (1 - interpol_par) + higher_val * interpol_par
    aver_err = lower_err * (1 - interpol_par) + higher_err * interpol_par
    print(f'Interpolated: {energy}, {aver_val}, {aver_err}')

    undressed_cs, new_stat_error, overall_sys_error = _undress_cross_section_measurement(
        alpha_0, aver_val, aver_err, cs, err_stat, err_sys, verbose=True
    )
    return energy, round(undressed_cs, 2), round(new_stat_error, 2), round(overall_sys_error, 2)


def plot_running_alpha():

    def _plot(x, y, err, title, x_axis_label, y_axis_label):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.errorbar(x, y, yerr=err, ecolor='black', color='red', fmt='.', markersize=0.5)

        plt.show()
        plt.close()

    _plot(*_read_data_running_alpha_real_parts(), title='Real(alpha)',
          x_axis_label='E[GeV]', y_axis_label='Re[alpha]')
    _plot(*_read_data_running_alpha_imaginary_parts(), title='Imaginary(alpha)',
          x_axis_label='E[GeV]', y_axis_label='Im[alpha]')
    _plot(*list(zip(*_get_running_alpha_data())), title='AlphaSquared',
          x_axis_label='E[GeV]', y_axis_label='|alpha|^2')


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')

    filepath_to_undress = '../data/raw_files/cmd_3_charged_kaons.csv'
    out_filepath = '../data/raw_files/cmd_3_charged_kaons_undressed.csv'

    running_alpha_data = _get_running_alpha_data()
    plot_running_alpha()

    converted = []

    with open(filepath_to_undress, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for energy, cs, stat_err, sys_err in reader:
            energy, cs, stat_err, sys_err = float(energy), float(cs), float(stat_err), float(sys_err)
            undressed_energy, undressed_cs, undressed_stat_err, undressed_sys_err = undress_data_point(
                energy, cs, stat_err, sys_err, alpha, running_alpha_data
            )
            print(f'{energy, cs, stat_err, sys_err} ->'
                  f' {undressed_energy, undressed_cs, undressed_stat_err, undressed_sys_err}')

            converted.append((undressed_energy, undressed_cs, undressed_stat_err, undressed_sys_err))

    with open(out_filepath, 'w') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(converted)
