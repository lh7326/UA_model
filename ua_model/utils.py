from typing import Callable, TypeVar

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')


def compose(f: Callable[[T], U], g: Callable[[S], T]) -> Callable[[S], U]:
    """
    A convenience function to carry out a function composition.

    Args:
        f (function):
        g (function):

    Returns:
        function: A new function corresponding to the composition of f and g.
                  I.e., compose(f, g)(x) = f(g(x)).

    """
    def h(x):
        return f(g(x))
    return h


def validate_branch_point_positions(t_0: float, t_in: float) -> None:
    """
    A helper function to validate the position of branch points.

    Here t_0 is the lowest branch point and t_in is a phenomenological constant
    that determines the position of the effective branch point that approximates
    the contribution of all the remaining branch points.

    We require the branch points to not be negative (t is off-the-mass-shell momentum
    squared in the +--- signature) and t_0 < t_in.

    Args:
        t_0 (float):
        t_in (float):

    Raises:
        ValueError

    """
    if t_0 < 0:
        raise ValueError(f'Negative t_0: {t_0}')
    if t_in < t_0:
        raise ValueError(f't_in must be larger than t_0!')
