import csv
import os.path
from collections import namedtuple
from typing import Tuple, List


Datapoint = namedtuple('Datapoint', 't is_charged')

DIR_NAME = '../data'


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

