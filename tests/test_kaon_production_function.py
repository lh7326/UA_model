from unittest import TestCase
import cmath

from kaon_production.function import function_kaon_cross_section
from kaon_production.data import Datapoint
from model_parameters import KaonParameters, KaonParametersSimplified


class TestKaonProductionFunction(TestCase):

    def test_function_cross_section__with_kaon_parameters(self):

        ts = [
            Datapoint(t=1.1230, is_charged=True),
            Datapoint(t=1.1230 + 0.73200j, is_charged=True),
            (0.73200j, 1.0),
            Datapoint(t=-0.73200j, is_charged=True),
            (1.1230 - 0.73200j, 1.0),
            (1.1230, 0.0),
            Datapoint(t=1.1230 + 0.73200j, is_charged=False),
            Datapoint(t=0.73200j, is_charged=False),
            (-0.73200j, 0.0),
            Datapoint(t=1.1230 - 0.73200j, is_charged=False),
        ]

        m_pion = 0.13957039
        parameters = KaonParameters(
            t_0_isoscalar=(9 * m_pion ** 2),
            t_0_isovector=(4 * m_pion ** 2),
            t_in_isoscalar=1.35,
            t_in_isovector=0.59,
            a_omega=0.004,
            mass_omega=0.78266,
            decay_rate_omega=0.00868,
            a_omega_prime=0.27,
            mass_omega_prime=1.410,
            decay_rate_omega_prime=0.29,
            a_omega_double_prime=0.0,
            mass_omega_double_prime=1.670,
            decay_rate_omega_double_prime=0.315,
            a_phi=-0.01,
            mass_phi=1.019461,
            decay_rate_phi=0.004249,
            a_phi_prime=0.0,
            mass_phi_prime=1.680,
            decay_rate_phi_prime=0.150,
            mass_phi_double_prime=2.159,
            decay_rate_phi_double_prime=0.137,
            a_rho=0.24,
            mass_rho=0.77525,
            decay_rate_rho=0.1474,
            a_rho_prime=0.1,
            mass_rho_prime=1.465,
            decay_rate_rho_prime=0.4,
            a_rho_double_prime=-0.1,
            mass_rho_double_prime=1.570,
            decay_rate_rho_double_prime=0.144,
            mass_rho_triple_prime=1.720,
            decay_rate_rho_triple_prime=0.25,
        )

        actual_values = function_kaon_cross_section(
            ts,
            k_meson_mass=0.493677,
            alpha=0.0072973525693,
            hc_squared=389379.3721,
            parameters=parameters,
        )

        expected_values = [
            1.422899583299445,
            7.322169806015914+5.145404160322535j,
            43.88590980666062-8.017152698541237j,
            43.88590980666064+8.01715269854124j,
            7.32216980601591-5.145404160322531j,
            0.5325913326692017,
            1.1155860768607209 + 0.7839399239773068j,
            1.5146068572452065 - 0.27669095858528914j,
            1.5146068572451994 + 0.27669095858528786j,
            1.115586076860725 - 0.7839399239773098j,
        ]

        for t, actual, expected in zip(ts, actual_values, expected_values):
            with self.subTest(msg=f't={t}'):
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

    def test_function_cross_section__with_kaon_parameters_simplified(self):

        ts = [
            Datapoint(t=1.1230, is_charged=True),
            Datapoint(t=1.1230 + 0.73200j, is_charged=True),
            (0.73200j, 1.0),
            Datapoint(t=-0.73200j, is_charged=True),
            Datapoint(t=1.1230 - 0.73200j, is_charged=True),
            (1.1230, 0.0),
            (1.1230 + 0.73200j, 0.0),
            (0.73200j, 0.0),
            (-0.73200j, 0.0),
            Datapoint(t=1.1230 - 0.73200j, is_charged=False),
        ]

        m_pion = 0.13957039
        parameters = KaonParametersSimplified(
            t_0_isoscalar=(9 * m_pion ** 2),
            t_0_isovector=(4 * m_pion ** 2),
            t_in_isoscalar=1.35,
            t_in_isovector=0.59,
            a_omega_prime=0.27,
            mass_omega_prime=1.410,
            decay_rate_omega_prime=0.29,
            a_omega_double_prime=0.0,
            mass_omega_double_prime=1.670,
            decay_rate_omega_double_prime=0.315,
            a_phi=-0.01,
            mass_phi=1.019461,
            decay_rate_phi=0.004249,
            a_phi_prime=0.0,
            mass_phi_prime=1.680,
            decay_rate_phi_prime=0.150,
            mass_phi_double_prime=2.159,
            decay_rate_phi_double_prime=0.137,
            a_rho_prime=0.1,
            mass_rho_prime=1.465,
            decay_rate_rho_prime=0.4,
            a_rho_double_prime=-0.1,
            mass_rho_double_prime=1.570,
            decay_rate_rho_double_prime=0.144,
            mass_rho_triple_prime=1.720,
            decay_rate_rho_triple_prime=0.25,
        )
        actual_values = function_kaon_cross_section(
            ts,
            k_meson_mass=0.493677,
            alpha=0.0072973525693,
            hc_squared=389379.3721,
            parameters=parameters,
        )

        expected_values = [
            2.721106392638103,
            14.002976557440112 + 9.840112390777117j,
            59.78991299967486 - 10.9225230709046j,
            59.78991299967489 + 10.922523070904605j,
            14.002976557440086 - 9.840112390777097j,
            0.08305375738667582,
            0.1432413591144838 + 0.10065796132075078j,
            0.029289844333965784 - 0.005350718615072836j,
            0.029289844333966465 + 0.00535071861507296j,
            0.1432413591144833 - 0.10065796132075043j,
        ]

        for t, actual, expected in zip(ts, actual_values, expected_values):
            with self.subTest(msg=f't={t}'):
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))
