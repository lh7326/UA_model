"""
This module contains some functions that will be useful in the construction of the model.

"""
import cmath
import math

from ua_model.function_utils import compose


def z_minus_its_reciprocal(z: complex) -> complex:
    """
    The function z - 1/z.

    I define this function explicitly because it plays an important role in the model.
    Furthermore, we also want to plot it and test it.

    Note: I do not handle the case of zero (or infinite) argument here.

    Args:
        z (complex): a nonzero complex number

    Returns:
        complex: The value of z - 1/z.

    """
    return z - (1 / z)


def square_root(z: complex) -> complex:
    """
    A branch of the square root. The cut is on the positive real axis; continuous from above.

    We need a branch of the square root other than the principal one. We choose the branch analytic
    on the complex plane minus the positive real numbers. I.e. z = r * exp(i * phi), r > 0, 0 <= phi < 2pi.
    Then square_root(r * exp(i*phi))= sqrt(r) * exp(i*phi/2), with 0 <= phi/2 < pi.

    Args:
        z (complex):

    Returns:
        complex: A square root of z.

    """
    r, phi = cmath.polar(z)
    # the original phi is from [-pi, pi]; but we do not want the branch cut on the negative real axis
    if phi < 0:
        phi = 2 * cmath.pi + phi
    return math.sqrt(r) * cmath.exp(1j * phi / 2.0)
