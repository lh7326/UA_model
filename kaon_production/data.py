import csv
import math
import os.path
from collections import namedtuple
from typing import Tuple, List


KaonDatapoint = namedtuple('KaonDatapoint', 't is_charged')

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


def read_data_new(file_name: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs = []
    ys = []
    stat_errs = []
    sys_errs = []

    filepath = os.path.join(DIR_NAME, 'new', file_name)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for x, y, stat_err, sys_err in reader:
            xs.append(float(x))
            ys.append(float(y))
            stat_errs.append(float(stat_err))
            sys_errs.append(float(sys_err))

    return xs, ys, stat_errs, sys_errs


def transform_energy_to_s(source_filepath: str, output_filepath: str) -> None:
    converted = []

    with open(source_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for energy, cross_section, statistical_error, systematic_error in reader:
            converted.append(
                (round(float(energy)**2, 6),
                 cross_section,
                 statistical_error,
                 systematic_error)
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


if __name__ == '__main__':
    file_name = 'snd_charged_kaons_undressed.csv'
    transform_energy_to_s(
        f'../data/raw_files/{file_name}',
        f'../data/new/{file_name}')
