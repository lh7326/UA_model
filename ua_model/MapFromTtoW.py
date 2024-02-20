import cmath
import math

from ua_model.functions import square_root
from ua_model.utils import validate_branch_point_positions


class MapFromTtoW:

    def __init__(self, t_0: float, t_in: float) -> None:
        """
        Initialize the coordinate map object.

        This object represents a function that maps from a four-sheeted Riemann surface
        of the complex t coordinate into the complex W-plane, in which we will construct the model.
        The coordinate t corresponds to the square of the (off-shell) momentum carried by the virtual photon.
        The physical sheet (also referred to as the first sheet) is mapped onto the left half of unit disk
        centered at the origin in the W-plane. It maps the upper half-plane onto the upper half of the semi-disk
        and the lower half-plane onto the lower half of the semi-disk.
        The second sheet is then mapped onto the right half of the unit disk,
        the third sheet onto the rest of the left half of the W-plane
        and the fourth sheet onto the rest of the right half-plane.

        Note that we use the signature +---, so t > 0 corresponds to time-like momenta.

        It will be convenient to decompose the mapping into two steps:
        1) q = sqrt_positive(t - t_0) and a = sqrt_positive(t_in - t_0).

            Here a > 0 is a constant that will be used in the next step.
            The function 'sqrt_positive' is the branch of the square root with the
            cut on the positive half of the real axis. That is, on its first sheet
            it is defined as
                   sqrt_positive(z) = sqrt(|z|) * exp(i * arg z / 2)  for 0 <= arg z < 2pi.

        2) W = i * ((a/q) - sqrt_negative(a - q) * sqrt_negative(a + q) / q).

            The function 'sqrt_negative' is the branch of the square root with the
            cut on the negative half of the real axis. That is, on its first sheet
            it is defined as
                   sqrt_negative(z) = sqrt(|z|) * exp(i * arg z / 2)  for -pi < arg z <= pi.

        It is important to note that we use different branches of the square root in steps 1 and 2!

        Args:
            t_0 (float): a positive number corresponding to the value of t at the lowest branch point
            t_in (float): a positive number larger than t_0 (a phenomenological constant)

        """
        self._validate_parameters(t_0, t_in)
        self.t_0 = t_0
        self.t_in = t_in
        self._a = math.sqrt(t_in - t_0)  # a numeric constant that is needed in the calculations

    def __call__(self, t: float) -> complex:
        """
        Return the value of W corresponding to a real argument and lying in the left half of the unit disk.
        That is, this is the map from the physical sheet.

        Args:
            t (float):

        Returns:
            complex

        """
        if t > self.t_0:  # we need to make sure that we are approaching the real axis from above
            t = t + 1e-15j
        q = square_root(t - self.t_0)  # the first sheet of the sqrt_positive
        # note that cmath.sqrt corresponds to the first sheet of sqrt_negative
        return 1j * ((self._a / q) - cmath.sqrt(self._a + q) * cmath.sqrt(self._a - q) / q)

    def map_from_sheet(self, t: complex, sheet: int):
        """
        Map from a specified sheet of the Riemann surface of t.
        Do not use for points lying directly in the cuts.

        Args:
            t (complex):
            sheet (int): The sheet number. (One of 1, 2, 3, 4.)

        Returns:
            complex

        """
        self._validate_sheet_number(sheet)
        if t.imag == 0 and t.real > self.t_0:
            raise ValueError('The value of t must be lying off the cut!')
        q = square_root(t - self.t_0)  # the first sheet of the sqrt_positive
        sign_1 = +1 if sheet in {1, 4} else -1
        sign_2 = +1 if sheet in {2, 4} else -1
        # note that cmath.sqrt corresponds to the first sheet of sqrt_negative
        return 1j * (
                sign_1 * (self._a / q) +
                sign_2 * cmath.sqrt(self._a + q) * cmath.sqrt(self._a - q) / q
        )

    @staticmethod
    def _validate_parameters(t_0, t_in):
        validate_branch_point_positions(t_0, t_in)

    @staticmethod
    def _validate_sheet_number(sheet: int):
        if not isinstance(sheet, int):
            raise ValueError('The value of "sheet" must be an integer between 1 and 4 (included)!')
        if not 1 <= sheet <= 4:
            raise ValueError(f'The allowed values for "sheet" are integers 1,2,3,4! Got {repr(sheet)}.')
