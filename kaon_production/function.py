from configparser import ConfigParser
from typing import List

from ua_model.KaonUAModel import KaonUAModel
from ua_model.KaonUAModelSimplified import KaonUAModelSimplified
from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from Task import Datapoint


def function_cross_section(
        ts: List[Datapoint],
        k_meson_mass: float,
        alpha: float,
        hc_squared: float,
        t_0_isoscalar: float,
        t_0_isovector: float,
        t_in_isoscalar: float,
        t_in_isovector: float,
        a_omega: float,
        mass_omega: float,
        decay_rate_omega: float,
        a_omega_prime: float,
        mass_omega_prime: float,
        decay_rate_omega_prime: float,
        a_omega_double_prime: float,
        mass_omega_double_prime: float,
        decay_rate_omega_double_prime: float,
        a_phi: float,
        mass_phi: float,
        decay_rate_phi: float,
        a_phi_prime: float,
        mass_phi_prime: float,
        decay_rate_phi_prime: float,
        mass_phi_double_prime: float,
        decay_rate_phi_double_prime: float,
        a_rho: float,
        mass_rho: float,
        decay_rate_rho: float,
        a_rho_prime: float,
        mass_rho_prime: float,
        decay_rate_rho_prime: float,
        a_rho_double_prime: float,
        mass_rho_double_prime: float,
        decay_rate_rho_double_prime: float,
        mass_rho_triple_prime: float,
        decay_rate_rho_triple_prime: float,
        ) -> List[complex]:

    ff_model = KaonUAModel(
        charged_variant=True,
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=t_in_isoscalar,
        t_in_isovector=t_in_isovector,
        a_omega=a_omega,
        a_omega_prime=a_omega_prime,
        a_omega_double_prime=a_omega_double_prime,
        a_phi=a_phi,
        a_phi_prime=a_phi_prime,
        a_rho=a_rho,
        a_rho_prime=a_rho_prime,
        a_rho_double_prime=a_rho_double_prime,
        mass_omega=mass_omega,
        decay_rate_omega=decay_rate_omega,
        mass_omega_prime=mass_omega_prime,
        decay_rate_omega_prime=decay_rate_omega_prime,
        mass_omega_double_prime=mass_omega_double_prime,
        decay_rate_omega_double_prime=decay_rate_omega_double_prime,
        mass_phi=mass_phi,
        decay_rate_phi=decay_rate_phi,
        mass_phi_prime=mass_phi_prime,
        decay_rate_phi_prime=decay_rate_phi_prime,
        mass_phi_double_prime=mass_phi_double_prime,
        decay_rate_phi_double_prime=decay_rate_phi_double_prime,
        mass_rho=mass_rho,
        decay_rate_rho=decay_rate_rho,
        mass_rho_prime=mass_rho_prime,
        decay_rate_rho_prime=decay_rate_rho_prime,
        mass_rho_double_prime=mass_rho_double_prime,
        decay_rate_rho_double_prime=decay_rate_rho_double_prime,
        mass_rho_triple_prime=mass_rho_triple_prime,
        decay_rate_rho_triple_prime=decay_rate_rho_triple_prime,
    )

    config = ConfigParser()
    config['constants'] = {'alpha': alpha, 'hc_squared': hc_squared}
    cross_section_model = ScalarMesonProductionTotalCrossSection(k_meson_mass, ff_model, config)

    results = []
    for datapoint in ts:
        if isinstance(datapoint, Datapoint):
            cross_section_model.form_factor.charged_variant = datapoint.is_charged
            results.append(cross_section_model(datapoint.t))
        else:
            cross_section_model.form_factor.charged_variant = bool(datapoint[1])
            results.append(cross_section_model(datapoint[0]))
    return results


# TODO: simplify passing arguments in this program --- use ModelParameters more often
# TODO: perhaps make factories: parameters -> models
def function_cross_section_simplified(
        ts: List[Datapoint],
        k_meson_mass: float,
        alpha: float,
        hc_squared: float,
        t_0_isoscalar: float,
        t_0_isovector: float,
        t_in_isoscalar: float,
        t_in_isovector: float,
        a_omega_prime: float,
        mass_omega_prime: float,
        decay_rate_omega_prime: float,
        a_omega_double_prime: float,
        mass_omega_double_prime: float,
        decay_rate_omega_double_prime: float,
        a_phi: float,
        mass_phi: float,
        decay_rate_phi: float,
        a_phi_prime: float,
        mass_phi_prime: float,
        decay_rate_phi_prime: float,
        mass_phi_double_prime: float,
        decay_rate_phi_double_prime: float,
        a_rho_prime: float,
        mass_rho_prime: float,
        decay_rate_rho_prime: float,
        a_rho_double_prime: float,
        mass_rho_double_prime: float,
        decay_rate_rho_double_prime: float,
        mass_rho_triple_prime: float,
        decay_rate_rho_triple_prime: float,
        ) -> List[complex]:

    ff_model = KaonUAModelSimplified(
        charged_variant=True,
        t_0_isoscalar=t_0_isoscalar,
        t_0_isovector=t_0_isovector,
        t_in_isoscalar=t_in_isoscalar,
        t_in_isovector=t_in_isovector,
        a_omega_prime=a_omega_prime,
        a_omega_double_prime=a_omega_double_prime,
        a_phi=a_phi,
        a_phi_prime=a_phi_prime,
        a_rho_prime=a_rho_prime,
        a_rho_double_prime=a_rho_double_prime,
        mass_omega_prime=mass_omega_prime,
        decay_rate_omega_prime=decay_rate_omega_prime,
        mass_omega_double_prime=mass_omega_double_prime,
        decay_rate_omega_double_prime=decay_rate_omega_double_prime,
        mass_phi=mass_phi,
        decay_rate_phi=decay_rate_phi,
        mass_phi_prime=mass_phi_prime,
        decay_rate_phi_prime=decay_rate_phi_prime,
        mass_phi_double_prime=mass_phi_double_prime,
        decay_rate_phi_double_prime=decay_rate_phi_double_prime,
        mass_rho_prime=mass_rho_prime,
        decay_rate_rho_prime=decay_rate_rho_prime,
        mass_rho_double_prime=mass_rho_double_prime,
        decay_rate_rho_double_prime=decay_rate_rho_double_prime,
        mass_rho_triple_prime=mass_rho_triple_prime,
        decay_rate_rho_triple_prime=decay_rate_rho_triple_prime,
    )

    config = ConfigParser()
    config['constants'] = {'alpha': alpha, 'hc_squared': hc_squared}
    cross_section_model = ScalarMesonProductionTotalCrossSection(k_meson_mass, ff_model, config)

    results = []
    for datapoint in ts:
        if isinstance(datapoint, Datapoint):
            cross_section_model.form_factor.charged_variant = datapoint.is_charged
            results.append(cross_section_model(datapoint.t))
        else:
            cross_section_model.form_factor.charged_variant = bool(datapoint[1])
            results.append(cross_section_model(datapoint[0]))
    return results
