"""
This is the model used Bianconi and Tomasi-Gustafsson as a background term when they were examining
nucleon form factor oscillations. This was originally proposed by Tomasi-Gustafsson and Rekalo.

Here the model is slightly adjusted by releasing the ``magic'' parameter 0.71 as a free parameter m_d.
(The symbol m_d stands for a "dipole mass".)

"""


class ETGMRModel:

    def __init__(self, a: float, m_a: float, m_d: float) -> None:
        self.a = a
        self.m_a_squared = m_a ** 2
        self.m_d_squared = m_d ** 2

    def __call__(self, s: complex) -> complex:
        return self.a / ((1 + s / self.m_a_squared) * ((1 - s / self.m_d_squared)**2))
