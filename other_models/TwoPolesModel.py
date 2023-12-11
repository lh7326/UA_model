"""
This is a model suggested by Bianconi and Tomasi-Gustafsson as a background term for examining
nucleon form factor oscillations. It was originally proposed by Brodsky and de Teramond.

"""


class TwoPolesModel:

    def __init__(self, a: float, m_1: float, m_2: float) -> None:
        self.a = a
        self.m_1_squared = m_1 ** 2
        self.m_2_squared = m_2 ** 2

    def __call__(self, s: complex) -> complex:
        return self.a / ((1 - s / self.m_1_squared) * (1 - s / self.m_2_squared))
