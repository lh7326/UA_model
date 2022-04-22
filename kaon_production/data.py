import csv
import os.path
from typing import Tuple, List

DIR_NAME = '../data'


def read_cross_section_data(file_name: str = 'kaon.csv') -> Tuple[List[float], List[float], List[float]]:
    ts = []
    form_factors = []
    sigmas = []

    filepath = os.path.join(DIR_NAME, file_name)
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for t, form_factor, sigma in reader:
            ts.append(float(t))
            form_factors.append(float(form_factor))
            sigmas.append(float(sigma))

    return ts, form_factors, sigmas
