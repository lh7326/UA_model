from configparser import ConfigParser
import csv
from kaon_production.data import read_data
import math


def _get_coefficient_function(config):
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')

    four_mass_squared = 4.0 * (kaon_mass ** 2)
    const = hc_squared * math.pi * (alpha**2) / 3.0

    def f(t):
        return (const / t) * ((1.0 - four_mass_squared / t) ** (3 / 2))
    return f


def error_for_square_root(val, err):
    return (1 / (2 * math.sqrt(val))) * err




if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')

    coefficient_f = _get_coefficient_function(config)

    charged_ts, charged_cross_sections_values, charged_errors = read_data(
        'charged_new_data2.csv')

    converted = []
    for t, cs, err in zip(charged_ts, charged_cross_sections_values, charged_errors):
        coeff = coefficient_f(t)
        ff = math.sqrt(cs / coeff)
        new_err = err / (2 * coeff * ff)
        converted.append((t, ff, new_err))

    with open('../data/charged_ff_2.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(converted)
