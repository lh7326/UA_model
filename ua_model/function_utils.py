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
