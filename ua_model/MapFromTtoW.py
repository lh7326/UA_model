import math

from ua_model.functions import square_root
from ua_model.utils import validate_branch_point_positions


class MapFromTtoW:

    def __init__(self, t_0: float, t_in: float) -> None:
        """
        TODO: add docstring

        Args:
            t_0:
            t_in:

        """
        self._validate_parameters(t_0, t_in)
        self.t_0 = t_0
        self.t_in = t_in
        self._a = math.sqrt(t_in - t_0)  # a numeric constant that is needed in the calculations

    def __call__(self, t: complex) -> complex:
        """
        TODO: add docstring

        Args:
            t:

        Returns:

        """
        z = square_root(t - self.t_0)
        transformed_z = (z + self._a) / (-z + self._a)  # the first Mobius transform
        v = square_root(transformed_z)
        return 1j * (v - 1) / (v + 1)  # the second Mobius transform

    @staticmethod
    def _validate_parameters(t_0, t_in):
        validate_branch_point_positions(t_0, t_in)
