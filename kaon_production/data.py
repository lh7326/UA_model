import csv
import os.path

DIR_NAME = '../data'


def read_cross_section_data(file_name='kaon.csv'):
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
