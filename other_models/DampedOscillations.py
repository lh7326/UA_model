import math


class DampedOscillations:

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: float) -> float:
        return self.a * math.exp(-1 * self.b * x) * math.cos(self.c * x + self.d)
