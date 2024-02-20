"""
Here I plot alternate definitions of the inverse coordinate map, from t to W.

"""
import cmath
from ua_model.MapFromTtoW import MapFromTtoW
from ua_model.functions import square_root

from plotting.plot_complex import plot_mapped_curves

T_0 = 1.0
T_IN = 2.0

map_from_t_to_w = MapFromTtoW(T_0, T_IN)


def t_to_w_version1(t):
    """
    The version used in the model.

    """
    return map_from_t_to_w.map_from_sheet(t, sheet=1)


def t_to_w_version2(t):
    """
    In this version I use the definition from the literature while using the alternate
    branch of the square root.

    The definition is:
    W = i * (sqrt(q_in + q) - sqrt(q_in - q)) / (sqrt(q_in + q) + sqrt(q_in - q)),
    where
    q_in = sqrt((t_in - t_0) / t_0)
    q = sqrt((t - t_0) / t_0)

    """
    q_in = square_root((T_IN - T_0) / T_0)
    q = square_root((t - T_0) / T_0)
    numerator = square_root(q_in + q) - square_root(q_in - q)
    denominator = square_root(q_in + q) + square_root(q_in - q)
    return 1j * numerator / denominator


def t_to_w_version3(t):
    """
    In this version I use the definition from the literature while using the principal
    branch of the square root.

    The definition is:
    W = i * (sqrt(q_in + q) - sqrt(q_in - q)) / (sqrt(q_in + q) + sqrt(q_in - q)),
    where
    q_in = sqrt((t_in - t_0) / t_0)
    q = sqrt((t - t_0) / t_0)

    """
    q_in = cmath.sqrt((T_IN - T_0) / T_0)
    q = cmath.sqrt((t - T_0) / T_0)
    numerator = cmath.sqrt(q_in + q) - cmath.sqrt(q_in - q)
    denominator = cmath.sqrt(q_in + q) + cmath.sqrt(q_in - q)
    return 1j * numerator / denominator


def plot_t_to_w(f):
    curves = [
        [(n * -0.1 + 1e-10j + T_0) for n in range(1, 1000)],  # between minus infinity and t_0
        [((T_IN - T_0) * n * 0.01 + 1e-10j + T_0) for n in range(1, 100)],  # between t_0 and t_in
        [(n * 0.1 + 1e-10j + T_IN) for n in range(1, 1000)],  # above t_in
        [(n * 0.1 - 20j) for n in range(-1000, 0)] + [(n * 0.1 - 20j) for n in range(1, 1000)],
        [n * (0.1 + 0.1j) for n in range(1, 1000)],
    ]

    mapped = [[f(z) for z in curve] for curve in curves]

    plot_mapped_curves('t-plane', 'W-plane', curves, mapped)


if __name__ == '__main__':
    from ua_model.MapFromWtoT import MapFromWtoT
    g = MapFromWtoT(T_0, T_IN)
    for t in [0.05, 1.7+1j, 8123-721j, -7.4+0.5j]:
        print(t, g(t_to_w_version1(t)), g(t_to_w_version2(t)), g(t_to_w_version3(t)))
    plot_t_to_w(t_to_w_version3)
