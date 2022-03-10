"""
TODO: add docstring

"""
import configparser

from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class KaonUAModel:
    """
    TODO: add docstring

    """
    def __init__(
            self,
            a_omega: float,
            a_omega_prime: float,
            a_omega_double_prime: float,
            a_phi: float,
            a_phi_prime: float,
            a_rho: float,
            a_rho_prime: float,
            t_in_isoscalar: float,
            t_in_isovector: float,
            config: configparser.ConfigParser,
            charged_variant: bool,
    ) -> None:
        """
        TODO: add docstring

        Args:
            a_omega:
            a_omega_prime:
            a_omega_double_prime:
            a_phi:
            a_phi_prime:
            a_rho:
            a_rho_prime:
            t_in_isoscalar:
            t_in_isovector:
            config:
            charged_variant:

        """
        # isoscalar components' proportionality constants
        self.a_omega = a_omega
        self.a_omega_prime = a_omega_prime
        self.a_omega_double_prime = a_omega_double_prime
        self.a_phi = a_phi
        self.a_phi_prime = a_phi_prime
        self.a_phi_double_prime = (
                0.5 - a_omega - a_omega_prime - a_omega_double_prime - a_phi - a_phi_prime
        )

        # isovector components' proportionality constants
        self.a_rho = a_rho
        self.a_rho_prime = a_rho_prime
        self.a_rho_double_prime = 0.5 - a_rho - a_rho_prime

        self.t_in_isoscalar = t_in_isoscalar
        self.t_in_isovector = t_in_isovector

        self._config = config
        self.charged_variant = charged_variant
        self._t_to_W_isoscalar = MapFromTtoW(
            t_0=config.getfloat('branch_points', 'kaon_isoscalar'),
            t_in=t_in_isoscalar,
        )
        self._t_to_W_isovector = MapFromTtoW(
            t_0=config.getfloat('branch_points', 'kaon_isovector'),
            t_in=t_in_isovector,
        )

        self._component_omega = None
        self._component_omega_prime = None
        self._component_omega_double_prime = None
        self._component_phi = None
        self._component_phi_prime = None
        self._component_phi_double_prime = None
        self._component_rho = None
        self._component_rho_prime = None
        self._component_rho_double_prime = None
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
            self.a_omega * self._component_omega(w)
            + self.a_omega_prime * self._component_omega_prime(w)
            + self.a_omega_double_prime * self._component_omega_double_prime(w)
            + self.a_phi * self._component_phi(w)
            + self.a_phi_prime * self._component_phi_prime(w)
            + self.a_phi_double_prime * self._component_phi_double_prime(w)
        )

    def _eval_isovector_contribution(self, t: complex) -> complex:
        w = self._t_to_W_isovector(t)
        return (
            self.a_rho * self._component_rho(w)
            + self.a_rho_prime * self._component_rho_prime(w)
            + self.a_rho_double_prime * self._component_rho_double_prime(w)
        )

    def _initialize_isoscalar_components(self) -> None:
        t_0 = self._config.getfloat('branch_points', 'kaon_isoscalar')

        # The construction below is perhaps a bit unfortunate, but it seems to me
        # less error-prone than explicitly repeating the same code section for each resonance.
        for resonance_name in [
            'omega', 'omega_prime', 'omega_double_prime',
            'phi', 'phi_prime', 'phi_double_prime',
        ]:
            mass = self._config.getfloat('resonances', resonance_name + '_mass')
            decay_rate = self._config.getfloat('resonances', resonance_name + '_decay_rate')
            component = self._build_component(t_0, self.t_in_isoscalar, mass, decay_rate)
            self.__setattr__('_component_' + resonance_name, component)

    def _initialize_isovector_components(self) -> None:
        t_0 = self._config.getfloat('branch_points', 'kaon_isovector')

        for resonance_name in ['rho', 'rho_prime', 'rho_double_prime']:
            mass = self._config.getfloat('resonances', resonance_name + '_mass')
            decay_rate = self._config.getfloat('resonances', resonance_name + '_decay_rate')
            component = self._build_component(t_0, self.t_in_isovector, mass, decay_rate)
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
