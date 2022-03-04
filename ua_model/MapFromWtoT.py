from ua_model.functions import z_minus_its_reciprocal
from ua_model.utils import validate_branch_point_positions


class MapFromWtoT:

    def __init__(self, t_0: float, t_in: float) -> None:
        """
        Initialize the coordinate map object.

        This object represents the function that maps from the complex w-plane
        in which we will construct the model onto the complex t-plane, where t
        is the square of the off-the-mass-shell momentum carried by the virtual photon.
        Note that we use the signature +---, so t^2 > 0 corresponds to time-like momenta.

        More precisely, the model is constructed on a four-sheeted Riemann surface, each sheet
        having the same t-coordinates but different W-coordinates. The map from W to t maps the complex plane
        onto itself in 4-to-1 fashion. The left half of the unit disk (center at the origin) is mapped
        onto the first (physical) sheet, the right half of the disk is mapped onto the second sheet and
        rest of the left half-plane and the rest of the right half-plane are mapped onto the remaining two sheets.

        The mapping is:
           t = t_0 - 4 * (t_in - t_0) / (W - 1/W)**2

           Here t_0 is the lowest branch point and t_in (t_in > t_0) is a phenomenological constant
           that determines the position of the second, effective, branch point.

        Args:
            t_0 (float): a positive number corresponding to the value of t at the lowest branch point
            t_in (float): a positive number larger than t_0 (a phenomenological constant)

        """
        self._validate_parameters(t_0, t_in)
        self.t_0 = t_0
        self.t_in = t_in
        self._t_in_minus_t_0 = t_in - t_0

    def __call__(self, w: complex) -> complex:
        """
        Returns the value of t corresponding to the argument.

        Args:
            w (complex):

        Returns:
            complex

        """
        return self.t_0 - 4.0 * self._t_in_minus_t_0 / (z_minus_its_reciprocal(w) ** 2)

    @staticmethod
    def _validate_parameters(t_0: float, t_in: float) -> None:
        validate_branch_point_positions(t_0, t_in)
