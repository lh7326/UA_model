from typing import Callable
from configparser import ConfigParser
import math


class NucleonPairToElectronPositronTotalCrossSection:

    def __init__(
            self,
            nucleon_mass: float,
            form_factor_model: Callable[[float], complex],
            config: ConfigParser
    ) -> None:
        """
        Initialize the calculator of the total cross-section for the process:
            nucleon + anti-nucleon -> electron + positron

        Args:
            nucleon_mass (float): the mass of the nucleon
            form_factor_model (callable): a model for the form factor
            config (ConfigParser): the configuration containing the values of the fine structure constant
                                   (under the key 'alpha'), and the square of the product of the reduced Planck
                                   constant and the speed of light, under the key 'hc_squared'

        """
        self.nucleon_mass = nucleon_mass
        self.form_factor = form_factor_model
        self.alpha = config.getfloat('constants', 'alpha')
        self.hc_squared = config.getfloat('constants', 'hc_squared')

        self._precalculated_coefficient_1 = self.hc_squared * 4 * math.pi * (self.alpha**2) / 3.0
        self._four_mass_squared = 4.0 * (self.nucleon_mass**2)

    def __call__(self, t: float) -> float:
        """
        #TODO: update!
        Evaluate the total cross-section for the process electron + positron -> meson + anti-meson.

        The cross-section depend on the corresponding form factor F through the formula:
            sigma = [(pi * alpha^2) / (3 * t)] * [1 - 4 * meson_mass^2 / t]^(3/2) * |F(t)|^2

        We express the cross-section in nanobarns.

        Args:
            t (float): the square of the four-momentum of the collision

        Returns:
            float: the value of the total cross-section in nanobarns

        """
        self.form_factor.electric = True
        electric_form_factor_modulus = abs(self.form_factor(t))
        self.form_factor.electric = False
        magnetic_form_factor_modulus = abs(self.form_factor(t))

        beta = math.sqrt(1.0 - self._four_mass_squared / abs(t))
        return (
            (self._precalculated_coefficient_1 * beta / t) *
            (magnetic_form_factor_modulus ** 2 + self._four_mass_squared * electric_form_factor_modulus**2 / (2 * t))
        )
