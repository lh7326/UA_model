"""
This is just for illustration and for comparison purposes. In particular, we will use this
implementation in unit tests of the U&A model.

"""


class VMDModel:

    def __init__(self) -> None:
        self._vector_mesons = []

    @classmethod
    def create(cls, vector_mesons: list) -> 'VMDModel':
        vmd_model = cls()
        for vector_meson in vector_mesons:
            vmd_model.add_vector_meson(vector_meson['coefficient'], vector_meson['mass'])
        return vmd_model

    def add_vector_meson(self, coefficient: float, mass: float) -> None:
        self._vector_mesons.append(
            {'mass': mass, 'coefficient': coefficient}
        )

    def __call__(self, t: complex) -> complex:
        if not self._vector_mesons:
            raise Exception('No vector mesons have been defined yet!')

        result = 0
        for vector_meson in self._vector_mesons:
            result += self._evaluate_meson_contribution(
                t, vector_meson['coefficient'], vector_meson['mass']
            )
        return result

    @staticmethod
    def _evaluate_meson_contribution(t: complex, coefficient: float, mass: float) -> complex:
        mass_squared = mass**2
        return coefficient * mass_squared / (mass_squared - t)
