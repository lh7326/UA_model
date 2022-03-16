from unittest import TestCase
import cmath

from ua_model.KaonUAModel import KaonUAModel


class TestKaonUAModel(TestCase):

    def test___call__(self):
        kaon_model = KaonUAModel(
            charged_variant=True,
            t_0_isoscalar=1.0,
            t_0_isovector=0.1,
            t_in_isoscalar=4.5,
            t_in_isovector=14.7,
            a_omega=0.21,
            a_omega_prime=0.09,
            a_omega_double_prime=0.12,
            a_phi=0.15,
            a_phi_prime=0.07,
            a_rho=0.34,
            a_rho_prime=0.03,
            a_rho_double_prime=0.09,
            mass_omega=1.4,
            decay_rate_omega=0.001,
            mass_omega_prime=1.5,
            decay_rate_omega_prime=0.001,
            mass_omega_double_prime=1.6,
            decay_rate_omega_double_prime=0.001,
            mass_phi=2.0,
            decay_rate_phi=0.01,
            mass_phi_prime=2.2,
            decay_rate_phi_prime=0.01,
            mass_phi_double_prime=2.4,
            decay_rate_phi_double_prime=0.01,
            mass_rho=3.1,
            decay_rate_rho=0.1,
            mass_rho_prime=3.2,
            decay_rate_rho_prime=0.1,
            mass_rho_double_prime=3.3,
            decay_rate_rho_double_prime=0.1,
            mass_rho_triple_prime=3.4,
            decay_rate_rho_triple_prime=0.1,
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
            {'t': 1.7, 'charged_variant': True, 'expected_value': 3.073269926471771-0.02695425107525793j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 1.8755035816025236+0.0038004114642766432j},
            {'t': 0.4+1.2j, 'charged_variant': True, 'expected_value': 0.9475875980415935+0.33143141222536704j},
            {'t': 162.42-0.647j, 'charged_variant': False,
             'expected_value': 0.027789176727690115+0.00011725561280149965j},
            {'t': 84.1-9124.1j, 'charged_variant': True,
             'expected_value': -5.691108792051046e-06-0.0006893509534412159j},
            {'t': 62.4j, 'charged_variant': False, 'expected_value': -0.012972244493240833-0.0646782195188189j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                print(actual, expected)
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
