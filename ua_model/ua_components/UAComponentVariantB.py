from ua_model.ua_components.UAComponent import UAComponent


class UAComponentVariantB(UAComponent):
    """
    A variant of the UA model component.

    This variant is used when the squared mass of the resonance lies above the t_in.
    The mapping from the physical sheet of t to W then maps the point t=mass**2
    somewhere on the upper left quarter of the unit circle.

    Expressing the VMD model in terms of the W-variable we find that it is possible
    to express the VMD model pole (for a resonance with zero width) in the t variable
    using the function
        (1)         [(W - W_pole) * (W + W_pole) * (W - 1/W_pole) * (W + 1/W_pole)]*(-1)
    of the W variable.

    The true pole of the resonance, however, corresponds to the point
    t = (mass - i * decay_width / 2)**2 (that is, the width is non-zero)
    and this does not lie on the unit circle in the W-plane.
    We want the shifted poles of the function of W-variable to satisfy the following conditions:
       a) they do not lie in the physical sheet
       b) they form complex conjugated pairs so that the resulting function satisfies the reality condition

    To satisfy this we modify the poles in (1) as follows:
      The pole W_pole will correspond to t=(mass - i * decay_width / 2)**2 and lie in the third sheet.
      (This is the sheet that neighbours the first sheet with the boundary being the left half of the unit circle.)

      Instead of the pole 1/W_pole that would now lie in the first sheet we use W_pole.conjugate().

      The pole -1 * W_pole lies in the fourth sheet and instead of -1/W_pole we use (-W_pole).conjugate().

      This way we obtain two conjugated pairs of poles, one pair in the third sheet and one pair in the fourth sheet.

    """

    def _evaluate_poles(self, meson_mass: float, meson_decay_rate: float) -> None:
        t = (meson_mass - 1j * meson_decay_rate / 2) ** 2

        w_1 = self.map_from_t_to_w.map_from_sheet(t, sheet=3)
        self.poles.extend([w_1, w_1.conjugate()])

        w_3 = self.map_from_t_to_w.map_from_sheet(t, sheet=4)
        self.poles.extend([w_3, w_3.conjugate()])

        assert len(self.poles) == 4
