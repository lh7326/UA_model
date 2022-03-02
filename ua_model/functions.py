"""
This module contains some functions that will be useful in the construction of the model.

"""
import cmath

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
