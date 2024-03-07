import warnings

from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class KaonUAModelPhiRatio:
    """
    Another U&A model for the charged or neutral kaon form factors.

    """
    def __init__(
            self,
            charged_variant: bool,
            t_0_isoscalar: float,
            t_in_isoscalar: float,
            t_0_isovector: float,
            t_in_isovector: float,
            a_omega: float,
            a_omega_double_prime: float,
            a_phi_charged: float,
            a_phi_neutral: float,
            a_phi_prime: float,
            a_rho: float,
            a_rho_prime: float,
            mass_omega: float,
            decay_rate_omega: float,
            mass_omega_double_prime: float,
            decay_rate_omega_double_prime: float,
            mass_phi: float,
            decay_rate_phi: float,
            mass_phi_prime: float,
            decay_rate_phi_prime: float,
            mass_phi_double_prime: float,
            decay_rate_phi_double_prime: float,
            mass_rho: float,
            decay_rate_rho: float,
            mass_rho_prime: float,
            decay_rate_rho_prime: float,
            mass_rho_double_prime: float,
            decay_rate_rho_double_prime: float,
    ) -> None:
        self.t_0_isoscalar = t_0_isoscalar
        self.t_in_isoscalar = t_in_isoscalar
        self.t_0_isovector = t_0_isovector
        self.t_in_isovector = t_in_isovector

        self.charged_variant = charged_variant

        # isoscalar components' proportionality constants
        self.a_omega = a_omega
        self.a_omega_double_prime = a_omega_double_prime
        self.a_phi_charged = a_phi_charged
        self.a_phi_neutral = a_phi_neutral
        self.a_phi_prime = a_phi_prime
        self.a_phi_double_prime_charged = (
                0.5 - a_omega - a_omega_double_prime - a_phi_charged - a_phi_prime
        )
        self.a_phi_double_prime_neutral = (
                0.5 - a_omega - a_omega_double_prime - a_phi_neutral - a_phi_prime
        )

        # isovector components' proportionality constants
        self.a_rho = a_rho
        self.a_rho_prime = a_rho_prime
        self.a_rho_double_prime = 0.5 - a_rho - a_rho_prime

        # read data about the resonances
        self.mass_omega = mass_omega
        self.decay_rate_omega = decay_rate_omega
        self.mass_omega_double_prime = mass_omega_double_prime
        self.decay_rate_omega_double_prime = decay_rate_omega_double_prime
        self.mass_phi = mass_phi
        self.decay_rate_phi = decay_rate_phi
        self.mass_phi_prime = mass_phi_prime
        self.decay_rate_phi_prime = decay_rate_phi_prime
        self.mass_phi_double_prime = mass_phi_double_prime
        self.decay_rate_phi_double_prime = decay_rate_phi_double_prime
        self.mass_rho = mass_rho
        self.decay_rate_rho = decay_rate_rho
        self.mass_rho_prime = mass_rho_prime
        self.decay_rate_rho_prime = decay_rate_rho_prime
        self.mass_rho_double_prime = mass_rho_double_prime
        self.decay_rate_rho_double_prime = decay_rate_rho_double_prime

        self._t_to_W_isoscalar = MapFromTtoW(
            t_0=self.t_0_isoscalar,
            t_in=self.t_in_isoscalar,
        )
        self._t_to_W_isovector = MapFromTtoW(
            t_0=self.t_0_isovector,
            t_in=self.t_in_isovector,
        )

        self._component_omega = None
        self._component_omega_double_prime = None
        self._component_phi = None
        self._component_phi_prime = None
        self._component_phi_double_prime = None
        self._component_rho = None
        self._component_rho_prime = None
        self._component_rho_double_prime = None
        self._initialize_isoscalar_components()
        self._initialize_isovector_components()

    def __call__(self, t: float) -> complex:
        isoscalar_contribution = self._eval_isoscalar_contribution(t)
        isovector_contribution = self._eval_isovector_contribution(t)

        if self.charged_variant:
            return isoscalar_contribution + isovector_contribution
        else:
            return isoscalar_contribution - isovector_contribution

    def _eval_isoscalar_contribution(self, t: float) -> complex:
        w = self._t_to_W_isoscalar(t)

        a_phi = self.a_phi_charged if self.charged_variant else self.a_phi_neutral
        a_phi_double_prime = (
            self.a_phi_double_prime_charged if self.charged_variant else self.a_phi_double_prime_neutral
        )

        return (
            self.a_omega * self._component_omega(w)
            + self.a_omega_double_prime * self._component_omega_double_prime(w)
            + a_phi * self._component_phi(w)
            + self.a_phi_prime * self._component_phi_prime(w)
            + a_phi_double_prime * self._component_phi_double_prime(w)
        )

    def _eval_isovector_contribution(self, t: float) -> complex:
        w = self._t_to_W_isovector(t)
        return (
            self.a_rho * self._component_rho(w)
            + self.a_rho_prime * self._component_rho_prime(w)
            + self.a_rho_double_prime * self._component_rho_double_prime(w)
        )

    def _initialize_isoscalar_components(self) -> None:
        # The construction below is perhaps a bit unfortunate, but it seems to me
        # less error-prone than explicitly repeating the same code section for each resonance.
        for resonance_name in [
            'omega', 'omega_double_prime',
            'phi', 'phi_prime', 'phi_double_prime',
        ]:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            component = self._build_component(self.t_0_isoscalar, self.t_in_isoscalar, mass, decay_rate)
            self.__setattr__('_component_' + resonance_name, component)

    def _initialize_isovector_components(self) -> None:
        for resonance_name in ['rho', 'rho_prime', 'rho_double_prime']:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            component = self._build_component(self.t_0_isovector, self.t_in_isovector, mass, decay_rate)
            self.__setattr__('_component_' + resonance_name, component)

    @staticmethod
    def _build_component(t_0: float, t_in: float, mass: float, decay_rate: float) -> UAComponent:
        t_meson_pole = (mass - 1j * decay_rate / 2) ** 2
        if mass**2 <= t_0:
            raise ValueError('Mass squared of the resonance must be above the t_0 threshold!')
        elif t_meson_pole.real < t_0:
            warnings.warn(f'The real part of the meson pole {t_meson_pole} lies below t_0!'
                          f'(mass = {mass}; width = {decay_rate})')
            return UAComponentVariantA(mass, decay_rate, MapFromTtoW(t_0, t_in))
        elif t_meson_pole.real < t_in:
            return UAComponentVariantA(mass, decay_rate, MapFromTtoW(t_0, t_in))
        else:
            return UAComponentVariantB(mass, decay_rate, MapFromTtoW(t_0, t_in))