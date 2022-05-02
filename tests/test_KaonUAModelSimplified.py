from unittest import TestCase
import cmath

from ua_model.KaonUAModelSimplified import KaonUAModelSimplified


class TestKaonUAModel(TestCase):

    def test___call__(self):
        kaon_model = KaonUAModelSimplified(
            charged_variant=True,
            t_0_isoscalar=1.0,
            t_0_isovector=0.1,
            t_in_isoscalar=4.5,
            t_in_isovector=14.7,
            a_omega_prime=0.21,
            a_omega_double_prime=0.12,
            a_phi=0.15,
            a_phi_prime=0.07,
            a_rho_prime=0.34,
            a_rho_double_prime=0.09,
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
            {'t': 1.7, 'charged_variant': True, 'expected_value': 2.1039108007279586-0.020192045540117224j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 0.9184510967906709+0.01007442756299409j},
            {'t': 0.4+1.2j, 'charged_variant': True, 'expected_value': 0.9763103569027347+0.2877442282086382j},
            {'t': 162.42-0.647j, 'charged_variant': False,
             'expected_value': 0.02733388745288759+0.00012166023857243473j},
            {'t': 84.1-9124.1j, 'charged_variant': True,
             'expected_value': -6.256333956869893e-06-0.0007606879308416056j},
            {'t': 62.4j, 'charged_variant': False, 'expected_value': -0.013664137093264453-0.06287169656749025j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))

    def test___call___2(self):
        m_pion = 0.13957039

        kaon_model = KaonUAModelSimplified(
            charged_variant=True,
            t_0_isoscalar=(9 * m_pion**2),
            t_0_isovector=(4 * m_pion**2),
            t_in_isoscalar=1.35,
            t_in_isovector=0.59,
            a_omega_prime=0.27,
            a_omega_double_prime=0.0,
            a_phi=-0.01,
            a_phi_prime=0.0,
            a_rho_prime=0.1,
            a_rho_double_prime=-0.1,
            mass_omega_prime=1.410,
            decay_rate_omega_prime=0.29,
            mass_omega_double_prime=1.670,
            decay_rate_omega_double_prime=0.315,
            mass_phi=1.019461,
            decay_rate_phi=0.004249,
            mass_phi_prime=1.680,
            decay_rate_phi_prime=0.150,
            mass_phi_double_prime=2.159,
            decay_rate_phi_double_prime=0.137,
            mass_rho_prime=1.465,
            decay_rate_rho_prime=0.4,
            mass_rho_double_prime=1.570,
            decay_rate_rho_double_prime=0.144,
            mass_rho_triple_prime=1.720,
            decay_rate_rho_triple_prime=0.25,
        )

        test_cases = [
            {'t': 1.1230, 'charged_variant': True, 'expected_value': 1.7105341828412286-0.10792678569739274j},
            {'t': 1.1230 + 0.73200j, 'charged_variant': True, 'expected_value': 1.4880965701557989+0.5712687887800826j},
            {'t': 0.73200j, 'charged_variant': True, 'expected_value': 0.9481530478048945+0.23311239372786224j},
            {'t': -0.73200j, 'charged_variant': True, 'expected_value': 0.9481530478048947-0.23311239372786227j},
            {'t': 1.1230 - 0.73200j, 'charged_variant': True, 'expected_value': 1.4880965701557973-0.5712687887800828j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
