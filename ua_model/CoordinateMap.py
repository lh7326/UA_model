from functions import z_minus_its_reciprocal


class CoordinateMap:

    def __init__(self, t_zero: float, t_in: float) -> None:
        """

        Args:
            t_zero:
            t_in:

        """
        self._validate_parameters(t_zero, t_in)
        self.t_zero = t_zero
        self.t_in = t_in
        self._t_in_minus_t_zero = t_in - t_zero

    def __call__(self, w: complex) -> complex:
        """

        Args:
            w:

        Returns:

        """
        return self.t_zero - 4.0 * self._t_in_minus_t_zero / (z_minus_its_reciprocal(w) ** 2)

    @staticmethod
    def _validate_parameters(t_zero: float, t_in: float) -> None:
        if not t_zero > 0:
            raise ValueError(f'Negative t_zero: {t_zero}')
        if t_in < t_zero:
            raise ValueError(f't_in must be larger than t_zero!')
