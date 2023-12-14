from configparser import ConfigParser
import csv
import math


def error_for_addition_uncorrelated(err1, err2):
    return math.sqrt(err1**2 + err2**2)


def error_for_addition_correlated(err1, err2):
    return err1 + err2


def error_for_multiplication_uncorrelated(val1, err1, val2, err2):
    rel_err1 = err1 / val1
    rel_err2 = err2 / val2
    return val1 * val2 * error_for_addition_uncorrelated(rel_err1, rel_err2)


def error_for_multiplication_correlated(val1, err1, val2, err2):
    return val1 * err2 + val2 * err1


def undress_cross_section_measurement(alpha_0, alpha_s_squared, alpha_squared_err, cs_val, cs_err_sys=0.0):
    factor = alpha_0**2 / alpha_s_squared
    factor_err = (factor / alpha_s_squared) * alpha_squared_err

    return cs_val * factor, error_for_multiplication_uncorrelated(factor, factor_err, cs_val, cs_err_sys)


def _get_running_alpha_data():
    filepath_real = '../data/running_alpha/real_parts.csv'
    filepath_imaginary = '../data/running_alpha/imaginary_parts.csv'

    energy_real = []
    alpha_real = []
    error_real = []
    with open(filepath_real, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for en, a_r, _, e_r, _ in reader:
            energy_real.append(float(en))
            alpha_real.append(float(a_r))
            error_real.append(float(e_r))

    energy_imaginary = []
    alpha_imaginary = []
    error_imaginary = []
    with open(filepath_imaginary, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for en, a_i, a_i_low, a_i_high in reader:
            energy_imaginary.append(float(en))
            alpha_imaginary.append(float(a_i))
            error_imaginary.append((float(a_i_high) - float(a_i_low)) / 2.0)

    energy, alpha_squared, error = [], [], []
    for en_r, en_i, a_r, a_i, er_r, er_i in zip(energy_real, energy_imaginary, alpha_real,
                                                alpha_imaginary, error_real, error_imaginary):
        assert en_r == en_i
        energy.append(en_r)
        alpha_squared.append(a_r**2 + a_i**2)
        error.append(2 * a_r * er_r + 2 * a_i * er_i)

    return zip(energy, alpha_squared, error)


def undress_data_point(energy, cs, err_stat, err_sys, alpha_0, running_alpha_data):
    running_alpha_data = list(running_alpha_data)
    len_data = len(running_alpha_data)
    i = 0
    running_alpha_data_point = running_alpha_data[i]
    if running_alpha_data_point[0] > energy:
        raise 'Running alpha data start at higher energies'
    while running_alpha_data_point[0] < energy:
        i += 1
        if i == len_data - 1:
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
    factor = (alpha_0**2) / aver_val
    print(factor)
    undressed_cs = cs * factor
    overall_sys_err = error_for_multiplication_correlated(factor, aver_err, cs, err_sys)
    return energy, undressed_cs, err_stat, overall_sys_err


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')

    filepath_to_undress = '../data/raw_files/snd_charged_kaons.csv'
    out_filepath = '../data/raw_files/snd_charged_kaons_undressed.csv'

    running_alpha_data = list(_get_running_alpha_data())

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
