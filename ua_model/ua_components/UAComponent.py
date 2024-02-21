"""
What is here called a U&A component corresponds (up to a proportionality factor) to the
contribution to the U&A model of a single vector meson resonance.
The complete U&A model is then a linear combination of all the components, each component corresponding
to a single resonance --- in an analogy with the VMD model.

The form of the components is basically the same except for one aspect that depends on the mass
of the resonance and on the position of the constants t_0 and t_in. One reason to separate the components
as individual objects is to allow us to conveniently select the appropriate form of the component
at the time of the initialization of the model.

"""
import operator
from abc import ABC, abstractmethod
import functools
from ua_model.MapFromTtoW import MapFromTtoW


class UAComponent(ABC):

    def __init__(self, meson_mass: float, meson_decay_rate: float, map_from_t_to_w: MapFromTtoW) -> None:
        self.w_n = map_from_t_to_w(0)
        self.map_from_t_to_w = map_from_t_to_w

        self._poles = []
        self._evaluate_poles(meson_mass, meson_decay_rate)

        # evaluate some further constants
        self._asymptotic_factor_denominator = (1 - (self.w_n ** 2)) ** 2
        self._resonant_factor_numerator = self._eval_resonant_factor_numerator()

    def __call__(self, w: complex) -> complex:
        """
        Evaluate the component at the given value of W.

        Args:
            w (complex):

        Returns:
            complex

        """
        return self._eval_asymptotic_factor(w) * self._eval_resonant_factor(w)

    def _eval_asymptotic_factor(self, w: complex) -> complex:
        # Note: This is shared by several components. Hence, it is not optimal to evaluate it here.
        # If needed, we may consider making a change here.
        return ((1 - w**2) ** 2) / self._asymptotic_factor_denominator

    def _eval_resonant_factor(self, w: complex) -> complex:
        denominator = functools.reduce(
            operator.mul,
            [w - pole for pole in self._poles],
            1,
        )
        return self._resonant_factor_numerator / denominator

    def _eval_resonant_factor_numerator(self) -> complex:
        if not len(self._poles) == 4:
            raise Exception('Poles have not been evaluated correctly yet!')

        return functools.reduce(
            operator.mul,
            [self.w_n - pole for pole in self._poles],
            1,
        )

    @abstractmethod
    def _evaluate_poles(self, meson_mass: float, meson_decay_rate: float) -> None:
        """
        This method should evaluate locations of the four poles in the W-plane
        and save them into the list self._poles.

        Args:
            meson_mass:
            meson_decay_rate:

        Returns:
            None

        """
        pass
