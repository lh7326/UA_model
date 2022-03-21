"""
This module contains a class to calculate the total cross-section for
the electromagnetic production of a scalar meson.

TODO: describe better
TODO: Add units (everywhere?)

"""
from typing import Callable
from configparser import ConfigParser
import math


class ScalarMesonProductionTotalCrossSection:

    def __init__(
            self,
            meson_mass: float,
            form_factor_model: Callable[[complex], complex],
            config: ConfigParser
    ) -> None:
        """

        Args:
            meson_mass (float): the mass of the scalar meson
            form_factor_model (callable): a model for the form factor
            config (ConfigParser): the configuration containing the required constants  TODO: specify!

        """
        self.meson_mass = meson_mass
        self.form_factor = form_factor_model
        self.alpha = config.getfloat('constants', 'alpha')
        self.hc_squared = config.getfloat('constants', 'hc_squared')

        self._precalculated_coefficient_1 = self.hc_squared * math.pi * (self.alpha**2) / 3.0
        self._four_mass_squared = 4.0 * (self.meson_mass**2)

    def __call__(self, t: complex) -> complex:
        """
        TODO: finish the docstring

        sigma = [(pi * alpha^2) / (3 * t)] * [1 - 4 * meson_mass^2 / t]^(3/2) * |F(t)|^2

        Args:
            t:

        Returns:

        """
        form_factor_modulus = abs(self.form_factor(t))

        return ((self._precalculated_coefficient_1 / t) *
                ((1.0 - self._four_mass_squared / t) ** (3/2)) *
                form_factor_modulus ** 2)
