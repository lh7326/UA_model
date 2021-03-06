from ua_model.ua_components.UAComponent import UAComponent
from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA
from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB
from ua_model.MapFromTtoW import MapFromTtoW


class SingleComponentModel:
    """
    A simple model containing only a single component. (I.e., considering only a single resonance.)

    """
    def __init__(self,
                 t_0: float,
                 t_in: float,
                 coupling_constant: float,
                 mass_resonance: float,
                 decay_rate_resonance: float) -> None:
        self.t_0 = t_0
        self.t_in = t_in
        self.coupling_constant = coupling_constant
        self.mass_resonance = mass_resonance
        self.decay_rate_resonance = decay_rate_resonance

        self._t_to_W = MapFromTtoW(t_0=self.t_0, t_in=self.t_in)
        self._component = self._build_component(self.t_0, self.t_in, self.mass_resonance, self.decay_rate_resonance)

    def __call__(self, t: complex) -> complex:
        if t.real < 0:
            raise ValueError('t must have a positive real part!')

        w = self._t_to_W(t)
        return self.coupling_constant * self._component(w)

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
