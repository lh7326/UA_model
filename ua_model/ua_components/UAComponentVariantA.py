from ua_model.ua_components.UAComponent import UAComponent


class UAComponentVariantA(UAComponent):
    """
    A variant of the UA model component.

    This variant is used when the squared mass of the resonance lies between
    the t_0 and t_in. The mapping from t to W then maps the point t=mass**2 somewhere
    into the segment of the imaginary axis between 0 and i. For such a value of W_meson
    the complex conjugate of W_meson equals minus W_meson. In this variant we use this identity
    to express (W + W_meson) and (W + 1/W_meson) in the VMD model as (W - W_meson.conjugate()), respectively
    (W - 1/W_meson.conjugate()). Afterwards we use this form even for the values of W_meson which do not lie
    on the imaginary axis.

    """
    def _eval_resonant_factor(self, w: complex) -> complex:
        denominator = (
            (w - self.w_meson) *
            (w - self.w_meson.conjugate()) *
            (w - 1 / self.w_meson) *
            (w - 1 / self.w_meson.conjugate())
        )
        return self._resonant_factor_numerator / denominator

    def _eval_resonant_factor_numerator(self) -> complex:
        return (
            (self.w_n - self.w_meson) *
            (self.w_n - self.w_meson.conjugate()) *
            (self.w_n - 1 / self.w_meson) *
            (self.w_n - 1 / self.w_meson.conjugate())
        )
