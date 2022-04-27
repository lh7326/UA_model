import os.path
from ModelParameters import Parameter


def report(directory, filename='grep.txt'):
    results = {}
    with open(os.path.join(directory, filename), 'r') as f:
        for line in f:
            parts = line.split(':')
            name = ':'.join(parts[:2])
            data = ':'.join(parts[2:]).strip(' \n')
            results[name] = eval(data)

    chi_data = [(key, val['chi_squared']) for key, val in results.items()]
    chi_data = sorted(chi_data, key=lambda p: p[1])
    print(len(chi_data))

    return chi_data


if __name__ == '__main__':
    directory = '/home/lukas/reports/pool5'
    print(report(directory))