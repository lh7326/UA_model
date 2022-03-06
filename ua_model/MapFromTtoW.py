import math

from ua_model.functions import square_root
from ua_model.utils import validate_branch_point_positions


class MapFromTtoW:

    def __init__(self, t_0: float, t_in: float) -> None:
        """
        Initialize the coordinate map object.

        This object represents the function that maps from the complex t-plane,
        where t is the square of the off-the-mass-shell momentum carried by the virtual photon,
        into a region of the complex W-plane, in which we will construct the model.
        Note that we use the signature +---, so t^2 > 0 corresponds to time-like momenta.

        More precisely, the function maps the t-plane onto the left half of the unit disk centered at the origin.
        It maps the upper half-plane onto the upper half of the semi-disk and the lower half-plane onto the lower half
        of the semi-disk.

        The mapping is:
           W = i * (sqrt(Tz) - 1) / (sqrt(Tz) + 1),
           where Tz =  (z + a) / (-z + a)
           and
           z = sqrt(t - t_0) and a = sqrt(t_in - t_0).
        All the square roots represent the branches defined on {z: 0 <= arg z < 2pi}
        as sqrt(z) = sqrt(|z|) * exp(i * arg z / 2).

        Args:
            t_0 (float): a positive number corresponding to the value of t at the lowest branch point
            t_in (float): a positive number larger than t_0 (a phenomenological constant)

        """
        self._validate_parameters(t_0, t_in)
        self.t_0 = t_0
        self.t_in = t_in
        self._a = math.sqrt(t_in - t_0)  # a numeric constant that is needed in the calculations

    def __call__(self, t: complex) -> complex:
        """
        Return the value of W corresponding to the argument and lying in the left half of the unit disk.

        Args:
            t (complex):

        Returns:
            complex

        """
        z = square_root(t - self.t_0)
        transformed_z = (z + self._a) / (-z + self._a)  # the first Mobius transform
        v = square_root(transformed_z)
        return 1j * (v - 1) / (v + 1)  # the second Mobius transform

    @staticmethod
    def _validate_parameters(t_0, t_in):
        validate_branch_point_positions(t_0, t_in)
