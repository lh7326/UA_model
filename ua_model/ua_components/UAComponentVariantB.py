from ua_model.ua_components.UAComponent import UAComponent


class UAComponentVariantB(UAComponent):
    """
    A variant of the UA model component.

    This variant is used when the squared mass of the resonance lies above the t_in.
    The mapping from t to W then maps the point t=mass**2 somewhere on the upper left quarter
    of the unit circle. For such a value of W_meson the complex conjugate of W_meson equals 1/W_meson.
    In this variant we use this identity to express (W + 1/W_meson) and (W - 1/W_meson) in the VMD model
    as (W + W_meson.conjugate()), respectively (W - W_meson.conjugate()).
    Afterwards we use this form even for the values of W_meson which do not lie on the unit circle.

    """
    def _eval_resonant_factor(self, w: complex) -> complex:
        denominator = (
            (w - self.w_meson) *
            (w - self.w_meson.conjugate()) *
            (w + self.w_meson) *
            (w + self.w_meson.conjugate())
        )
        return self._resonant_factor_numerator / denominator

    def _eval_resonant_factor_numerator(self) -> complex:
        return (
            (self.w_n - self.w_meson) *
            (self.w_n - self.w_meson.conjugate()) *
            (self.w_n + self.w_meson) *
            (self.w_n + self.w_meson.conjugate())
        )
