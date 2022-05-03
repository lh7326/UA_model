from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class KaonUAModelSimplified:
    """
    Another U&A model for the charged or neutral kaon form factors. We don not consider omega
    and rho resonances, since their contributions are small due to their masses being significantly lower
    than the lowest mass of two kaons.

    """
    def __init__(
            self,
            charged_variant: bool,
            t_0_isoscalar: float,
            t_in_isoscalar: float,
            t_0_isovector: float,
            t_in_isovector: float,
            a_omega_prime: float,
            a_omega_double_prime: float,
            a_phi: float,
            a_phi_prime: float,
            a_rho_prime: float,
            a_rho_double_prime: float,
            mass_omega_prime: float,
            decay_rate_omega_prime: float,
            mass_omega_double_prime: float,
            decay_rate_omega_double_prime: float,
            mass_phi: float,
            decay_rate_phi: float,
            mass_phi_prime: float,
            decay_rate_phi_prime: float,
            mass_phi_double_prime: float,
            decay_rate_phi_double_prime: float,
            mass_rho_prime: float,
            decay_rate_rho_prime: float,
            mass_rho_double_prime: float,
            decay_rate_rho_double_prime: float,
            mass_rho_triple_prime: float,
            decay_rate_rho_triple_prime: float,
    ) -> None:
        self.t_0_isoscalar = t_0_isoscalar
        self.t_in_isoscalar = t_in_isoscalar
        self.t_0_isovector = t_0_isovector
        self.t_in_isovector = t_in_isovector

        self.charged_variant = charged_variant

        # isoscalar components' proportionality constants
        self.a_omega_prime = a_omega_prime
        self.a_omega_double_prime = a_omega_double_prime
        self.a_phi = a_phi
        self.a_phi_prime = a_phi_prime
        self.a_phi_double_prime = (
                0.5 - a_omega_prime - a_omega_double_prime - a_phi - a_phi_prime
        )

        # isovector components' proportionality constants
        self.a_rho_prime = a_rho_prime
        self.a_rho_double_prime = a_rho_double_prime
        self.a_rho_triple_prime = 0.5 - a_rho_prime - a_rho_double_prime

        # read data about the resonances
        self.mass_omega_prime = mass_omega_prime
        self.decay_rate_omega_prime = decay_rate_omega_prime
        self.mass_omega_double_prime = mass_omega_double_prime
        self.decay_rate_omega_double_prime = decay_rate_omega_double_prime
        self.mass_phi = mass_phi
        self.decay_rate_phi = decay_rate_phi
        self.mass_phi_prime = mass_phi_prime
        self.decay_rate_phi_prime = decay_rate_phi_prime
        self.mass_phi_double_prime = mass_phi_double_prime
        self.decay_rate_phi_double_prime = decay_rate_phi_double_prime
        self.mass_rho_prime = mass_rho_prime
        self.decay_rate_rho_prime = decay_rate_rho_prime
        self.mass_rho_double_prime = mass_rho_double_prime
        self.decay_rate_rho_double_prime = decay_rate_rho_double_prime
        self.mass_rho_triple_prime = mass_rho_triple_prime
        self.decay_rate_rho_triple_prime = decay_rate_rho_triple_prime

        self._t_to_W_isoscalar = MapFromTtoW(
            t_0=self.t_0_isoscalar,
            t_in=self.t_in_isoscalar,
        )
        self._t_to_W_isovector = MapFromTtoW(
            t_0=self.t_0_isovector,
            t_in=self.t_in_isovector,
        )

        self._component_omega_prime = None
        self._component_omega_double_prime = None
        self._component_phi = None
        self._component_phi_prime = None
        self._component_phi_double_prime = None
        self._component_rho_prime = None
        self._component_rho_double_prime = None
        self._component_rho_triple_prime = None
        self._initialize_isoscalar_components()
        self._initialize_isovector_components()

    def __call__(self, t: complex) -> complex:
        if t.real < 0:
            raise ValueError('t must have a positive real part!')

        isoscalar_contribution = self._eval_isoscalar_contribution(t)
        isovector_contribution = self._eval_isovector_contribution(t)

        if self.charged_variant:
            return isoscalar_contribution + isovector_contribution
        else:
            return isoscalar_contribution - isovector_contribution

    def _eval_isoscalar_contribution(self, t: complex) -> complex:
        w = self._t_to_W_isoscalar(t)
        return (
            self.a_omega_prime * self._component_omega_prime(w)
            + self.a_omega_double_prime * self._component_omega_double_prime(w)
            + self.a_phi * self._component_phi(w)
            + self.a_phi_prime * self._component_phi_prime(w)
            + self.a_phi_double_prime * self._component_phi_double_prime(w)
        )

    def _eval_isovector_contribution(self, t: complex) -> complex:
        w = self._t_to_W_isovector(t)
        return (
            self.a_rho_prime * self._component_rho_prime(w)
            + self.a_rho_double_prime * self._component_rho_double_prime(w)
            + self.a_rho_triple_prime * self._component_rho_triple_prime(w)
        )

    def _initialize_isoscalar_components(self) -> None:
        # The construction below is perhaps a bit unfortunate, but it seems to me
        # less error-prone than explicitly repeating the same code section for each resonance.
        for resonance_name in [
            'omega_prime', 'omega_double_prime',
            'phi', 'phi_prime', 'phi_double_prime',
        ]:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            component = self._build_component(self.t_0_isoscalar, self.t_in_isoscalar, mass, decay_rate)
            self.__setattr__('_component_' + resonance_name, component)

    def _initialize_isovector_components(self) -> None:
        for resonance_name in ['rho_prime', 'rho_double_prime', 'rho_triple_prime']:
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
