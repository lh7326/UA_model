from unittest import TestCase
import cmath

from ua_model.KaonUAModelB import KaonUAModelB


class TestKaonUAModelB(TestCase):

    def test___call__(self):
        kaon_model = KaonUAModelB(
            charged_variant=True,
            t_0_isoscalar=1.0,
            t_0_isovector=0.1,
            t_in_isoscalar=4.5,
            t_in_isovector=14.7,
            a_omega=0.21,
            a_omega_double_prime=0.12,
            a_phi=0.15,
            a_phi_prime=0.07,
            a_rho=0.34,
            a_rho_prime=0.03,
            mass_omega=1.4,
            decay_rate_omega=0.001,
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
            {'t': 1.7, 'charged_variant': True, 'expected_value': 2.8480184741889722+0.026333786578943648j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 1.6356180204063044-0.004733935106687839j},
            {'t': 0.4, 'charged_variant': True, 'expected_value': 1.1194085226018666+0.005265350018094823j},
            {'t': 162.42, 'charged_variant': False,
             'expected_value': 0.02283307753591677+3.3959304439700073e-06j},
            {'t': 162.42, 'charged_variant': True,
             'expected_value': -0.03998839126561036+3.3959304438872953e-06j},
            {'t': 84.1, 'charged_variant': True,
             'expected_value': -0.08142168854369632+9.425543634006539e-06j},
            {'t': 62.4, 'charged_variant': False, 'expected_value': 0.06828166252483078+1.5112423578746648e-05j},
            {'t': -0.583, 'charged_variant': True, 'expected_value': 0.8757462444764337},
            {'t': -0.583, 'charged_variant': False, 'expected_value': -0.06089372640851898},
            {'t': -42.9173, 'charged_variant': True, 'expected_value': 0.12063573873105779},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
