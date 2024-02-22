from unittest import TestCase
import cmath

from common.utils import function_cross_section, function_form_factor_or_cross_section
from kaon_production.data import KaonDatapoint
from model_parameters import KaonParameters, KaonParametersSimplified

# TODO: extend!


class TestCommonUtils(TestCase):

    def test_function_cross_section__with_kaon_parameters(self):

        ts = [
            KaonDatapoint(t=1.1230, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=0.73200, is_charged=True, is_for_cross_section=True),
            (0.73200, 1.0, 1.0),
            KaonDatapoint(t=-0.73200, is_charged=True, is_for_cross_section=True),
            (1.1230, 1.0, 1.0),
            (1.1230, 0.0, 1.0),
            KaonDatapoint(t=1.1230, is_charged=False, is_for_cross_section=True),
            KaonDatapoint(t=0.73200, is_charged=False, is_for_cross_section=True),
            (-0.73200, 0.0, 1.0),
            KaonDatapoint(t=-1.1230, is_charged=False, is_for_cross_section=True),
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

        actual_values = function_cross_section(
            ts,
            product_particle_mass=0.493677,
            alpha=0.0072973525693,
            hc_squared=389379.3721,
            parameters=parameters,
        )

        expected_values = [
            1.3305954557093416,
            15.598313561047673,
            15.598313561047673,
            3.066292705763645,
            1.3305954557093416,
            0.9603086503862627,
            0.9603086503862627,
            13.135748921343643,
            0.028065803623050833,
            0.005781174539189479,
        ]

        for t, actual, expected in zip(ts, actual_values, expected_values):
            with self.subTest(msg=f't={t}'):
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test "cross section/form factor" datapoint validation'):
            with self.assertRaises(ValueError):
                function_cross_section(
                    [KaonDatapoint(t=1.1230, is_charged=True, is_for_cross_section=False)],
                    product_particle_mass=0.493677,
                    alpha=0.0072973525693,
                    hc_squared=389379.3721,
                    parameters=parameters,
                )

    def test_function_cross_section__with_kaon_parameters_simplified(self):

        ts = [
            KaonDatapoint(t=1.1230, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=-1.1230, is_charged=True, is_for_cross_section=True),
            (0.73200, 1.0, 1.0),
            KaonDatapoint(t=-0.73200, is_charged=True, is_for_cross_section=True),
            (1.1230, 0.0, 1.0),
            (-1.1230, 0.0, 1.0),
            (0.73200, 0.0, 1.0),
            (-0.73200, 0.0, 1.0),
            KaonDatapoint(t=1.1230, is_charged=False, is_for_cross_section=True),
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
        actual_values = function_cross_section(
            ts,
            product_particle_mass=0.493677,
            alpha=0.0072973525693,
            hc_squared=389379.3721,
            parameters=parameters,
        )

        expected_values = [
            3.4234397922515987,
            0.48390598244875044,
            11.053068468257893,
            3.609259003645634,
            0.0476603610075804,
            0.0001769877297064973,
            0.03820243830713717,
            0.0007060717666394876,
            0.0476603610075804,
        ]

        for t, actual, expected in zip(ts, actual_values, expected_values):
            with self.subTest(msg=f't={t}'):
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

    def test_function_form_factor_or_cross_section__with_kaon_parameters(self):

        ts = [
            KaonDatapoint(t=1.1230, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=-1.1230, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=1.1230, is_charged=True, is_for_cross_section=False),
            (0.73200, 1.0, 1.0),
            KaonDatapoint(t=-0.73200, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=-0.73200, is_charged=True, is_for_cross_section=False),
            (1.1230, 1.0, 1.0),
            (-1.1230, 1.0, 0.0),
            (1.1230, 0.0, 1.0),
            (-1.1230, 0.0, 0.0),
            KaonDatapoint(t=1.1230, is_charged=False, is_for_cross_section=True),
            KaonDatapoint(t=0.73200, is_charged=False, is_for_cross_section=True),
            KaonDatapoint(t=-0.73200, is_charged=False, is_for_cross_section=False),
            (-0.73200, 0.0, 1.0),
            (-0.73200, 0.0, 0.0),
            KaonDatapoint(t=0.21, is_charged=False, is_for_cross_section=True),
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

        actual_values = function_form_factor_or_cross_section(
            ts,
            product_particle_mass=0.493677,
            alpha=0.0072973525693,
            hc_squared=389379.3721,
            parameters=parameters,
        )

        expected_values = [
            1.3305954557093416,
            0.39617959667831565,
            1.1985185677395396,
            15.598313561047673,
            3.066292705763645,
            0.7354479312680484,
            1.3305954557093416,
            0.6539850212443172,
            0.9603086503862627,
            0.07900049078778032,
            0.9603086503862627,
            13.135748921343643,
            0.07036130192534568,
            0.028065803623050833,
            0.07036130192534568,
            3.6192741050998296,
        ]

        for t, actual, expected in zip(ts, actual_values, expected_values):
            with self.subTest(msg=f't={t}'):
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))
