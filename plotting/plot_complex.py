import cmath
import random

import matplotlib.pyplot as plt
import matplotlib.colors

from ua_model.functions import square_root, z_minus_its_reciprocal
from ua_model.MapFromWtoT import MapFromWtoT


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


def plot_w_to_t(t_0, t_in, number_of_angles=3):
    curves_inside = [
        [base * n / 100 for n in range(50, 100)] for base in
        [cmath.exp(2j * cmath.pi * k / number_of_angles) for k in range(1, number_of_angles + 1)]
    ]
    curves_outside = [
        [base * (n + 50) / 50 for n in range(1, 100)] for base in
        [cmath.exp(2j * cmath.pi * k / number_of_angles) for k in range(number_of_angles)]
    ]
    curves = curves_inside + curves_outside

    f = MapFromWtoT(t_0, t_in)
    mapped = [[f(z) for z in curve] for curve in curves]

    plot_mapped_curves('W-plane', 't-plane', curves, mapped)


if __name__ == '__main__':
    plot_z_minus_its_reciprocal()

