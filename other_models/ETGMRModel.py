"""
This is the model used Bianconi and Tomasi-Gustafsson as a background term when they were examining
nucleon form factor oscillations. This was originally proposed by Tomasi-Gustafsson and Rekalo.

"""


class ETGMRModel:

    def __init__(self, a: float, m_a: float) -> None:
        self.a = a
        self.m_a_squared = m_a ** 2

    def __call__(self, s: complex) -> complex:
        return self.a / ((1 + s / self.m_a_squared) * ((1 - s / 0.71)**2))
