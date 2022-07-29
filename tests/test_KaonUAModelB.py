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
            {'t': 1.7, 'charged_variant': True, 'expected_value': 2.833713686104681-0.026098804207375276j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 1.6349283545517954+0.004620037297262777j},
            {'t': 0.4+1.2j, 'charged_variant': True, 'expected_value': 0.9626535493160868+0.3024515794728611j},
            {'t': 162.42-0.647j, 'charged_variant': False,
             'expected_value': 0.0255536766560733+0.00011344168647315351j},
            {'t': 84.1-9124.1j, 'charged_variant': True,
             'expected_value': -5.962543662602656e-06-0.0007201950671674711j},
            {'t': 62.4j, 'charged_variant': False, 'expected_value': -0.012181619593386869-0.05928607568642251j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))

    def test___call___2(self):
        kaon_model = KaonUAModelB(
            charged_variant=True,
            t_0_isoscalar=0.17531851630709158,
            t_0_isovector=0.07791934058092959,
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
            {'t': 1.7, 'charged_variant': True, 'expected_value': 2.8367557064873146-0.027743617685714858j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 1.6371740678666105+0.0031724417146562368j},
            {'t': 0.4+1.2j, 'charged_variant': True, 'expected_value': 0.9635319901700389+0.302257376647137j},
            {'t': 162.42-0.647j, 'charged_variant': False,
             'expected_value': 0.025559278162859647+0.00011346805059342094j},
            {'t': 84.1-9124.1j, 'charged_variant': True,
             'expected_value': -5.967699168267497e-06-0.0007208147331877005j},
            {'t': 62.4j, 'charged_variant': False, 'expected_value': -0.012187220864615354-0.05929700651742506j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
