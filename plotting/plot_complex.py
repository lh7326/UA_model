import cmath
import random

import matplotlib.pyplot as plt
import matplotlib.colors

from ua_model.functions import square_root, z_minus_its_reciprocal
from ua_model.MapFromWtoT import MapFromWtoT
from ua_model.MapFromTtoW import MapFromTtoW


def plot_mapped_curves(domain_title, range_title, domain_curves, mapped_curves):

    assert len(domain_curves) == len(mapped_curves)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(domain_title)
    ax2.set_title(range_title)

    colors = list(matplotlib.colors.CSS4_COLORS)
    random.shuffle(colors)
    colors = colors[:len(domain_curves)]
    for curve, mapped, color in zip(domain_curves, mapped_curves, colors):
        _plot_curve(ax1, curve, color)
        _plot_curve(ax2, mapped, color)
    plt.show()


def _plot_curve(axes, curve, color):
    x = [z.real for z in curve]
    y = [z.imag for z in curve]
    axes.plot(x, y, color=color)


def plot_square_root():
    curves_outward = [
        [base * n / 20 for n in range(1, 100)] for base in
        [cmath.exp(2j * cmath.pi * k / 3) for k in range(3)]
    ]
    curves_circles = [
        [r * cmath.exp(2j * cmath.pi * k / 100) for k in range(1, 100)] for r in [0.1, 1, 5]
    ]
    curves = curves_outward + curves_circles
    mapped = [[square_root(z) for z in curve] for curve in curves]

    plot_mapped_curves('z', 'sqrt(z)', curves, mapped)


def plot_z_minus_its_reciprocal(number_of_angles=5):
    curves_inside = [
        [base * n / 100 for n in range(10, 99)] for base in
        [cmath.exp(2j * cmath.pi * k / number_of_angles) for k in range(number_of_angles)]
    ]
    curves_outside = [
        [base * (n + 10) / 10 for n in range(1, 100)] for base in
        [cmath.exp(2j * cmath.pi * k / number_of_angles) for k in range(number_of_angles)]
    ]
    curves = curves_inside + curves_outside

    mapped = [[z_minus_its_reciprocal(z) for z in curve] for curve in curves]

    plot_mapped_curves('z', 'z - 1/z', curves, mapped)


def plot_w_to_t(t_0, t_in):
    bases = [
        cmath.exp(1j * cmath.pi * 2 / 3),
        cmath.exp(1j * cmath.pi * 5 / 6),
        cmath.exp(1j * cmath.pi * 8 / 7),
    ]
    curves_inside = [[base * n / 1000 for n in range(100, 990)] for base in bases]
    curves_outside = [[base * (n + 100) / 100 for n in range(10, 1000)] for base in bases]
    curves = curves_inside + curves_outside

    f = MapFromWtoT(t_0, t_in)
    mapped = [[f(z) for z in curve] for curve in curves]

    plot_mapped_curves('W-plane', 't-plane', curves, mapped)


def plot_t_to_w(t_0, t_in):
    curves = [
        [(n * -0.1 + t_0) for n in range(1, 1000)],  # between minus infinity and t_0
        [((t_in - t_0) * n * 0.01 + t_0) for n in range(1, 100)],  # between t_0 and t_in
        [(n * 0.1 + t_in) for n in range(1, 1000)],  # above t_in
        [(n * -0.1 + 54j) for n in range(1, 1000)],
        [(n * 0.1 + 54j) for n in range(1, 1000)],
        [(n * 0.1 - 20j) for n in range(-1000, 0)] + [(n * 0.1 - 20j) for n in range(1, 1000)],
        [n * (0.1 + 0.1j) for n in range(1, 1000)],
        [(n * 0.1j + 17) for n in range(1, 1000)],
    ]

    f = MapFromTtoW(t_0, t_in)
    mapped = [[f(z) for z in curve] for curve in curves]

    plot_mapped_curves('t-plane', 'W-plane', curves, mapped)


if __name__ == '__main__':
    plot_t_to_w(t_0=1, t_in=2)
