from unittest import TestCase
import cmath

from ua_model.KaonUAModelPhiRatio import KaonUAModelPhiRatio


class TestKaonUAModelPhiRatio(TestCase):

    def test___call__(self):
        kaon_model = KaonUAModelPhiRatio(
            charged_variant=True,
            t_0_isoscalar=1.0,
            t_0_isovector=0.1,
            t_in_isoscalar=4.5,
            t_in_isovector=14.7,
            a_omega=0.21,
            a_omega_double_prime=0.12,
            a_phi_charged=0.15,
            a_phi_neutral=-0.72,
            a_phi_prime=0.07,
            a_rho=0.34,
            a_rho_prime=0.09,
            mass_omega=1.5,
            decay_rate_omega=0.001,
            mass_omega_double_prime=1.6,
            decay_rate_omega_double_prime=0.001,
            mass_phi=2.0,
            decay_rate_phi=0.01,
            mass_phi_prime=2.2,
            decay_rate_phi_prime=0.01,
            mass_phi_double_prime=2.4,
            decay_rate_phi_double_prime=0.01,
            mass_rho=3.2,
            decay_rate_rho=0.1,
            mass_rho_prime=3.3,
            decay_rate_rho_prime=0.1,
            mass_rho_double_prime=3.4,
            decay_rate_rho_double_prime=0.1,
        )

        with self.subTest(msg='Testing "charged_variant" flag'):
            self.assertTrue(kaon_model.charged_variant)

        with self.subTest(msg='Test normalization (t=0) --- the variant K+ K-'):
            expected = 1.0
            kaon_model.charged_variant = True
            actual = kaon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected))

        with self.subTest(msg='Test normalization (t=0) --- the variant K0 anti-K0'):
            expected = 0.0
            kaon_model.charged_variant = False
            actual = kaon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))

        test_cases = [
            {'t': 1.7, 'charged_variant': True, 'expected_value': 2.1165682374379453+0.020414280898780467j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 0.6245673749213958-0.02536587787590667j},
            {'t': 0.4, 'charged_variant': True, 'expected_value': 1.1098076991968817+0.005281909928996892j},
            {'t': 162.42, 'charged_variant': False,
             'expected_value': 0.013785166591677594+5.868966908498406e-05j},
            {'t': 84.1, 'charged_variant': True,
             'expected_value': -0.08608936078598757+9.425543634011589e-06j},
            {'t': 62.4, 'charged_variant': False, 'expected_value': 0.0429809394203983+0.0002712457631785561j},
            {'t': -0.23, 'charged_variant': False, 'expected_value': -0.011447142407457145},
            {'t': -0.23, 'charged_variant': True, 'expected_value': 0.948249279932925},
            {'t': -101.7621, 'charged_variant': False, 'expected_value': -0.0171260153213613},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
