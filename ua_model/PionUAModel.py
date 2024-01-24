from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class PionUAModel:
    """
    The U&A model for the charged pion form factor.

    """
    def __init__(
            self,
            t_0_isovector: float,
            t_in_isovector: float,
            a_rho_prime: float,
            a_rho_double_prime: float,
            a_rho_triple_prime: float,
            mass_rho: float,
            decay_rate_rho: float,
            mass_rho_prime: float,
            decay_rate_rho_prime: float,
            mass_rho_double_prime: float,
            decay_rate_rho_double_prime: float,
            mass_rho_triple_prime: float,
            decay_rate_rho_triple_prime: float,
            w_pole: complex,
            w_zero: complex,
    ) -> None:
        self.t_0_isovector = t_0_isovector
        self.t_in_isovector = t_in_isovector

        self.w_pole = w_pole
        self.w_zero = w_zero

        # isovector components' proportionality constants
        self.a_rho_prime = a_rho_prime
        self.a_rho_double_prime = a_rho_double_prime
        self.a_rho_triple_prime = a_rho_triple_prime
        self.a_rho = 0.5 - a_rho_prime - a_rho_double_prime - a_rho_triple_prime

        # read data about the resonances
        self.mass_rho = mass_rho
        self.decay_rate_rho = decay_rate_rho
        self.mass_rho_prime = mass_rho_prime
        self.decay_rate_rho_prime = decay_rate_rho_prime
        self.mass_rho_double_prime = mass_rho_double_prime
        self.decay_rate_rho_double_prime = decay_rate_rho_double_prime
        self.mass_rho_triple_prime = mass_rho_triple_prime
        self.decay_rate_rho_triple_prime = decay_rate_rho_triple_prime

        self._t_to_W_isovector = MapFromTtoW(
            t_0=self.t_0_isovector,
            t_in=self.t_in_isovector,
        )
        self.w_n = self._t_to_W_isovector(0)

        self._component_rho = None
        self._component_rho_prime = None
        self._component_rho_double_prime = None
        self._component_rho_triple_prime = None
        self._initialize_isovector_components()

    def __call__(self, t: complex) -> complex:
        return self._eval_additional_factor(t) * self._eval_isovector_contribution(t)

    def _eval_isovector_contribution(self, t: complex) -> complex:
        w = self._t_to_W_isovector(t)
        return (
            self.a_rho * self._component_rho(w)
            + self.a_rho_prime * self._component_rho_prime(w)
            + self.a_rho_double_prime * self._component_rho_double_prime(w)
            + self.a_rho_triple_prime * self._component_rho_triple_prime(w)
        )

    def _eval_additional_factor(self, t: complex) -> complex:
        w = self._t_to_W_isovector(t)
        zero_factor = (w - self.w_zero) / (self.w_n - self.w_zero)
        pole_factor = (self.w_n - self.w_pole) / (w - self.w_pole)
        return zero_factor * pole_factor

    def _initialize_isovector_components(self) -> None:
        for resonance_name in ['rho', 'rho_prime', 'rho_double_prime', 'rho_triple_prime']:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            component = self._build_component(self.t_0_isovector, self.t_in_isovector, mass, decay_rate)
            self.__setattr__('_component_' + resonance_name, component)

    @staticmethod
    def _build_component(t_0: float, t_in: float, mass: float, decay_rate: float) -> UAComponent:
        map_from_t_to_w = MapFromTtoW(t_0, t_in)
        w_n = map_from_t_to_w(0)
        t_meson_pole = (mass - 1j * decay_rate / 2) ** 2
        w_meson = map_from_t_to_w(t_meson_pole)

        mass_squared = mass**2
        if mass_squared < t_0:
            raise ValueError('Mass squared of the resonance must be above the t_0 threshold!')
        elif mass_squared < t_in:
            return UAComponentVariantA(w_n, w_meson)
        else:
            return UAComponentVariantB(w_n, w_meson)
