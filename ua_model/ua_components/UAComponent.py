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
from abc import ABC, abstractmethod


class UAComponent(ABC):

    def __init__(self, w_n: complex, w_meson: complex) -> None:
        self.w_n = w_n
        self.w_meson = w_meson

        # evaluate some constants
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

    @abstractmethod
    def _eval_resonant_factor(self, w: complex) -> complex:
        """
        This method should return the value of the appropriate resonant factor at the given W.

        NOTE: It is expected that this method will access the (constant) attribute '_resonant_factor_numerator'.

        Different variant of the model components differ by small details in this section of the function form.
        We decide on which variant to apply based on the value of the mass of the resonance and t_0 and t_in.

        Args:
            w (complex):

        Returns:
            complex

        """
        pass

    @abstractmethod
    def _eval_resonant_factor_numerator(self) -> complex:
        """
        This method should evaluate the numerator of the resonant factor.
        It is called only during the initialization of the component. (The numerator is a constant.)

        Returns:
            complex

        """
        pass
