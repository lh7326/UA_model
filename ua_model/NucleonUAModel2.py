from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class NucleonUAModel2:
    """
    The U&A model for the proton and neutron form factors.

    """
    def __init__(
            self,
            proton: bool,
            electric: bool,
            mass: float,
            magnetic_moment_proton: float,
            magnetic_moment_neutron: float,
            t_0_dirac_isoscalar: float,
            t_in_dirac_isoscalar: float,
            t_0_dirac_isovector: float,
            t_in_dirac_isovector: float,
            t_0_pauli_isoscalar: float,
            t_in_pauli_isoscalar: float,
            t_0_pauli_isovector: float,
            t_in_pauli_isovector: float,
            a_dirac_omega: float,
            a_dirac_omega_prime: float,
            a_dirac_phi: float,
            a_dirac_phi_prime: float,
            a_dirac_rho: float,
            a_pauli_omega: float,
            a_pauli_phi: float,
            a_pauli_phi_prime: float,
            mass_omega: float,
            decay_rate_omega: float,
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
            mass_rho: float,
            decay_rate_rho: float,
            mass_rho_prime: float,
            decay_rate_rho_prime: float,
            mass_rho_double_prime: float,
            decay_rate_rho_double_prime: float,
    ) -> None:
        self.proton = proton
        self.electric = electric
        self.mass = mass
        self.magnetic_moment_proton = magnetic_moment_proton
        self.magnetic_moment_neutron = magnetic_moment_neutron

        self.t_0_dirac_isoscalar = t_0_dirac_isoscalar
        self.t_in_dirac_isoscalar = t_in_dirac_isoscalar
        self.t_0_dirac_isovector = t_0_dirac_isovector
        self.t_in_dirac_isovector = t_in_dirac_isovector
        self.t_0_pauli_isoscalar = t_0_pauli_isoscalar
        self.t_in_pauli_isoscalar = t_in_pauli_isoscalar
        self.t_0_pauli_isovector = t_0_pauli_isovector
        self.t_in_pauli_isovector = t_in_pauli_isovector

        # read data about the resonances
        self.mass_omega = mass_omega
        self.decay_rate_omega = decay_rate_omega
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
        self.mass_rho = mass_rho
        self.decay_rate_rho = decay_rate_rho
        self.mass_rho_prime = mass_rho_prime
        self.decay_rate_rho_prime = decay_rate_rho_prime
        self.mass_rho_double_prime = mass_rho_double_prime
        self.decay_rate_rho_double_prime = decay_rate_rho_double_prime

        # Dirac isoscalar form factor components' coupling constants
        self.a_dirac_omega = a_dirac_omega
        self.a_dirac_omega_prime = a_dirac_omega_prime
        self.a_dirac_phi = a_dirac_phi
        self.a_dirac_phi_prime = a_dirac_phi_prime

        # Dirac isovector form factor components' coupling constants
        self.a_dirac_rho = a_dirac_rho

        # Pauli isoscalar form factor components' coupling constants
        self.a_pauli_omega = a_pauli_omega
        self.a_pauli_phi = a_pauli_phi
        self.a_pauli_phi_prime = a_pauli_phi_prime

        self._t_to_W_dirac_isoscalar = MapFromTtoW(
            t_0=self.t_0_dirac_isoscalar,
            t_in=self.t_in_dirac_isoscalar,
        )
        self._t_to_W_dirac_isovector = MapFromTtoW(
            t_0=self.t_0_dirac_isovector,
            t_in=self.t_in_dirac_isovector,
        )
        self._t_to_W_pauli_isoscalar = MapFromTtoW(
            t_0=self.t_0_pauli_isoscalar,
            t_in=self.t_in_pauli_isoscalar,
        )
        self._t_to_W_pauli_isovector = MapFromTtoW(
            t_0=self.t_0_pauli_isovector,
            t_in=self.t_in_pauli_isovector,
        )

        self._dirac_component_omega = None
        self._dirac_component_omega_prime = None
        self._dirac_component_omega_double_prime = None
        self._dirac_component_phi = None
        self._dirac_component_phi_prime = None
        self._dirac_component_phi_double_prime = None
        self._dirac_component_rho = None
        self._dirac_component_rho_prime = None
        self._dirac_component_rho_double_prime = None
        self._pauli_component_omega = None
        self._pauli_component_omega_prime = None
        self._pauli_component_omega_double_prime = None
        self._pauli_component_phi = None
        self._pauli_component_phi_prime = None
        self._pauli_component_phi_double_prime = None
        self._pauli_component_rho = None
        self._pauli_component_rho_prime = None
        self._pauli_component_rho_double_prime = None
        self._initialize_isoscalar_components(name='dirac')
        self._initialize_isovector_components(name='dirac')
        self._initialize_isoscalar_components(name='pauli')
        self._initialize_isovector_components(name='pauli')

    def __call__(self, t: complex) -> complex:
        if t.real < 0:
            raise ValueError('t must have a positive real part!')

        dirac_isoscalar_contribution = self._eval_dirac_isoscalar_contribution(t)
        dirac_isovector_contribution = self._eval_dirac_isovector_contribution(t)
        pauli_isoscalar_contribution = self._eval_pauli_isoscalar_contribution(t)
        pauli_isovector_contribution = self._eval_pauli_isovector_contribution(t)

        if self.proton:
             dirac_form_factor = dirac_isoscalar_contribution + dirac_isovector_contribution
             pauli_form_factor = pauli_isoscalar_contribution + pauli_isovector_contribution
        else:
            dirac_form_factor = dirac_isoscalar_contribution - dirac_isovector_contribution
            pauli_form_factor = pauli_isoscalar_contribution - pauli_isovector_contribution

        if self.electric:
            return dirac_form_factor + (t / (4 * self.mass ** 2)) * pauli_form_factor
        else:
            return dirac_form_factor + pauli_form_factor

    # TODO: refactor!
    def _eval_dirac_isoscalar_contribution(self, t: complex) -> complex:
        w = self._t_to_W_dirac_isoscalar(t)
        return (
            0.5 * self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_double_prime(w) +
            self.a_dirac_omega_prime * (
                self._dirac_component_phi_double_prime(w) * self._dirac_component_omega_prime(w) *
                (self.mass_phi_double_prime**2 - self.mass_omega_prime**2) / (
                        self.mass_phi_double_prime**2 - self.mass_omega_double_prime**2) +
                self._dirac_component_omega_double_prime(w) * self._dirac_component_omega_prime(w) *
                (self.mass_omega_double_prime ** 2 - self.mass_omega_prime ** 2) / (
                        self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2) -
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_double_prime(w)
            ) +
            self.a_dirac_phi_prime * (
                self._dirac_component_phi_double_prime(w) * self._dirac_component_phi_prime(w) *
                (self.mass_phi_double_prime ** 2 - self.mass_phi_prime ** 2) / (
                        self.mass_phi_double_prime ** 2 - self.mass_omega_double_prime ** 2) +
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_prime(w) *
                (self.mass_omega_double_prime ** 2 - self.mass_phi_prime ** 2) / (
                        self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2) -
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_double_prime(w)
            ) +
            self.a_dirac_omega * (
                self._dirac_component_phi_double_prime(w) * self._dirac_component_omega(w) *
                (self.mass_phi_double_prime ** 2 - self.mass_omega ** 2) / (
                        self.mass_phi_double_prime ** 2 - self.mass_omega_double_prime ** 2) +
                self._dirac_component_omega_double_prime(w) * self._dirac_component_omega(w) *
                (self.mass_omega_double_prime ** 2 - self.mass_omega ** 2) / (
                        self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2) -
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_double_prime(w)
            ) +
            self.a_dirac_phi * (
                self._dirac_component_phi_double_prime(w) * self._dirac_component_phi(w) *
                (self.mass_phi_double_prime ** 2 - self.mass_phi ** 2) / (
                        self.mass_phi_double_prime ** 2 - self.mass_omega_double_prime ** 2) +
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi(w) *
                (self.mass_omega_double_prime ** 2 - self.mass_phi ** 2) / (
                        self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2) -
                self._dirac_component_omega_double_prime(w) * self._dirac_component_phi_double_prime(w)
            )
        )

    def _eval_dirac_isovector_contribution(self, t: complex) -> complex:
        w = self._t_to_W_dirac_isovector(t)
        return (
            0.5 * self._dirac_component_rho_prime(w) * self._dirac_component_rho_double_prime(w) +
            self.a_dirac_rho * (
                self._dirac_component_rho(w) * self._dirac_component_rho_prime(w) *
                (self.mass_rho_prime**2 - self.mass_rho**2) / (
                    self.mass_rho_prime**2 - self.mass_rho_double_prime**2) +
                self._dirac_component_rho(w) * self._dirac_component_rho_double_prime(w) *
                (self.mass_rho_double_prime ** 2 - self.mass_rho ** 2) / (
                    self.mass_rho_double_prime ** 2 - self.mass_rho_prime ** 2) -
                self._dirac_component_rho_prime(w) * self._dirac_component_rho_double_prime(w)
            )
        )

    def _eval_pauli_isoscalar_contribution(self, t: complex) -> complex:
        w = self._t_to_W_pauli_isoscalar(t)
        norm = 0.5 * (self.magnetic_moment_proton + self.magnetic_moment_neutron - 1.0)
        return (
            norm * self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
            self._pauli_component_omega_prime(w) +
            self.a_pauli_phi_prime * (
                self._pauli_component_phi_double_prime(w) * self._pauli_component_phi_prime(w) *
                self._pauli_component_omega_prime(w) *
                ((self.mass_phi_double_prime**2 - self.mass_phi_prime**2) /
                 (self.mass_phi_double_prime**2 - self.mass_omega_double_prime**2)) *
                ((self.mass_omega_prime ** 2 - self.mass_phi_prime ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_omega_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_omega_prime(w) *
                self._pauli_component_phi_prime(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_phi_prime ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2)) *
                ((self.mass_omega_prime ** 2 - self.mass_phi_prime ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_phi_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_phi_prime(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_phi_prime ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_omega_prime ** 2)) *
                ((self.mass_phi_double_prime ** 2 - self.mass_phi_prime ** 2) /
                 (self.mass_phi_double_prime ** 2 - self.mass_omega_prime ** 2)) -
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_omega_prime(w)
            ) +
            self.a_pauli_omega * (
                self._pauli_component_phi_double_prime(w) * self._pauli_component_omega_prime(w) *
                self._pauli_component_omega(w) *
                ((self.mass_phi_double_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_phi_double_prime ** 2 - self.mass_omega_double_prime ** 2)) *
                ((self.mass_omega_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_omega_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_omega_prime(w) *
                self._pauli_component_omega(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2)) *
                ((self.mass_omega_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_phi_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_omega(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_omega_prime ** 2)) *
                ((self.mass_phi_double_prime ** 2 - self.mass_omega ** 2) /
                 (self.mass_phi_double_prime ** 2 - self.mass_omega_prime ** 2)) -
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_omega_prime(w)
            ) +
            self.a_pauli_phi * (
                self._pauli_component_phi_double_prime(w) * self._pauli_component_omega_prime(w) *
                self._pauli_component_phi(w) *
                ((self.mass_phi_double_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_phi_double_prime ** 2 - self.mass_omega_double_prime ** 2)) *
                ((self.mass_omega_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_omega_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_omega_prime(w) *
                self._pauli_component_phi(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_phi_double_prime ** 2)) *
                ((self.mass_omega_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_omega_prime ** 2 - self.mass_phi_double_prime ** 2)) +
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_phi(w) *
                ((self.mass_omega_double_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_omega_double_prime ** 2 - self.mass_omega_prime ** 2)) *
                ((self.mass_phi_double_prime ** 2 - self.mass_phi ** 2) /
                 (self.mass_phi_double_prime ** 2 - self.mass_omega_prime ** 2)) -
                self._pauli_component_omega_double_prime(w) * self._pauli_component_phi_double_prime(w) *
                self._pauli_component_omega_prime(w)
            )
        )

    def _eval_pauli_isovector_contribution(self, t: complex) -> complex:
        w = self._t_to_W_pauli_isovector(t)
        norm = 0.5 * (self.magnetic_moment_proton - self.magnetic_moment_neutron - 1.0)
        return (norm * self._pauli_component_rho(w) * self._pauli_component_rho_prime(w) *
                self._pauli_component_rho_double_prime(w))

    def _initialize_isoscalar_components(self, name) -> None:
        # The construction below is perhaps a bit unfortunate, but it seems to me
        # less error-prone than explicitly repeating the same code section for each resonance.
        for resonance_name in [
            'omega', 'omega_prime', 'omega_double_prime',
            'phi', 'phi_prime', 'phi_double_prime',
        ]:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            t_0 = self.__getattribute__(f't_0_{name}_isoscalar')
            t_in = self.__getattribute__(f't_in_{name}_isoscalar')
            component = self._build_component(t_0, t_in, mass, decay_rate)
            self.__setattr__(f'_{name}_component_' + resonance_name, component)

    def _initialize_isovector_components(self, name) -> None:
        for resonance_name in ['rho', 'rho_prime', 'rho_double_prime']:
            mass = self.__getattribute__('mass_' + resonance_name)
            decay_rate = self.__getattribute__('decay_rate_' + resonance_name)
            t_0 = self.__getattribute__(f't_0_{name}_isovector')
            t_in = self.__getattribute__(f't_in_{name}_isovector')
            component = self._build_component(t_0, t_in, mass, decay_rate)
            self.__setattr__(f'_{name}_component_' + resonance_name, component)

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
