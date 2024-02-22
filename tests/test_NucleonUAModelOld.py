from unittest import TestCase
import cmath

from ua_model.NucleonUAModelOld import NucleonUAModelOld


class TestNucleonUAModelOld(TestCase):

    def test___call__(self):
        nucleon_model = NucleonUAModelOld(
            proton=True,
            electric=True,
            nucleon_mass=1.0,
            magnetic_moment_proton=0.3,
            magnetic_moment_neutron=1.7,
            t_0_dirac_isoscalar=1.0,
            t_0_dirac_isovector=0.1,
            t_in_dirac_isoscalar=4.5,
            t_in_dirac_isovector=14.7,
            t_0_pauli_isoscalar=1.1,
            t_0_pauli_isovector=0.2,
            t_in_pauli_isoscalar=3.9,
            t_in_pauli_isovector=14.5,
            a_dirac_omega=0.21,
            a_dirac_omega_prime=0.09,
            a_dirac_phi=0.15,
            a_dirac_phi_prime=0.07,
            a_dirac_rho=0.34,
            a_pauli_omega=-0.17,
            a_pauli_phi=-0.42,
            a_pauli_phi_prime=0.09,
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
        )

        with self.subTest(msg='Test normalization (t=0) --- proton electric'):
            expected = 1.0
            nucleon_model.electric = True
            nucleon_model.proton = True
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- neutron electric'):
            expected = 0.0
            nucleon_model.electric = True
            nucleon_model.proton = False
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- proton magnetic'):
            expected = 0.3
            nucleon_model.electric = False
            nucleon_model.proton = True
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- neutron magnetic'):
            expected = 1.7
            nucleon_model.electric = False
            nucleon_model.proton = False
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test asymptotic behaviour --- neutron magnetic'):
            nucleon_model.electric = False
            nucleon_model.proton = False
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- proton magnetic'):
            nucleon_model.electric = False
            nucleon_model.proton = True
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- neutron electric'):
            nucleon_model.electric = True
            nucleon_model.proton = False
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- proton electric'):
            nucleon_model.electric = True
            nucleon_model.proton = True
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        test_cases = [
            {'t': 1.7, 'proton': True, 'electric': False,
             'expected_value': 9.361329307240192-0.07904453193348238j},
            {'t': 1.7, 'proton': False, 'electric': True,
             'expected_value': 6.475078254240392+0.05867853865854828j},
            {'t': 0.4, 'proton': True, 'electric': True,
             'expected_value': 1.1290430688305544+0.007583331687804014j},
            {'t': 162.42, 'proton': False, 'electric': False,
             'expected_value': -0.002231655477265035-5.855302267986061e-07j},
            {'t': 84.1, 'proton': True, 'electric': True,
             'expected_value': 0.06887750469995232-4.9575912210333375e-06j},
            {'t': -62.4, 'proton': False, 'electric': False,
             'expected_value': -0.004895982501977045},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                nucleon_model.proton = case['proton']
                nucleon_model.electric = case['electric']
                actual = nucleon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))

    def test___call___2(self):
        nucleon_model = NucleonUAModelOld(
            proton=True,
            electric=True,
            nucleon_mass=0.938272,
            magnetic_moment_proton=2.79,
            magnetic_moment_neutron=-1.91,
            t_0_dirac_isoscalar=0.17531904388276887,
            t_0_dirac_isovector=0.07791957505900839,
            t_in_dirac_isoscalar=0.9001581776629138,
            t_in_dirac_isovector=2.713739494786232,
            t_0_pauli_isoscalar=0.17531904388276887,
            t_0_pauli_isovector=0.07791957505900839,
            t_in_pauli_isoscalar=1.0512202460515163,
            t_in_pauli_isovector=4.176812892690669,
            a_dirac_omega=1.3200505056850964,
            a_dirac_omega_prime=0.11215926509622014,
            a_dirac_phi=-1.013845791442124,
            a_dirac_phi_prime=0.26717864421535276,
            a_dirac_rho=0.06479104707666819,
            a_pauli_omega=-0.32774864723704805,
            a_pauli_phi=0.07195092152400877,
            a_pauli_phi_prime=0.349116996120233,
            mass_omega=0.78266,
            decay_rate_omega=0.00868,
            mass_omega_prime=1.41,
            decay_rate_omega_prime=0.29,
            mass_omega_double_prime=1.67,
            decay_rate_omega_double_prime=0.315,
            mass_phi=1.019461,
            decay_rate_phi=0.004249,
            mass_phi_prime=1.68,
            decay_rate_phi_prime=0.15,
            mass_phi_double_prime=2.159,
            decay_rate_phi_double_prime=0.137,
            mass_rho=0.77526,
            decay_rate_rho=0.1474,
            mass_rho_prime=1.465,
            decay_rate_rho_prime=0.4,
            mass_rho_double_prime=1.72,
            decay_rate_rho_double_prime=0.25,
        )

        with self.subTest(msg='Test normalization (t=0) --- proton electric'):
            expected = 1.0
            nucleon_model.electric = True
            nucleon_model.proton = True
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- neutron electric'):
            expected = 0.0
            nucleon_model.electric = True
            nucleon_model.proton = False
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- proton magnetic'):
            expected = 2.79
            nucleon_model.electric = False
            nucleon_model.proton = True
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test normalization (t=0) --- neutron magnetic'):
            expected = -1.91
            nucleon_model.electric = False
            nucleon_model.proton = False
            actual = nucleon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))

        with self.subTest(msg='Test asymptotic behaviour --- neutron magnetic'):
            nucleon_model.electric = False
            nucleon_model.proton = False
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- proton magnetic'):
            nucleon_model.electric = False
            nucleon_model.proton = True
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- neutron electric'):
            nucleon_model.electric = True
            nucleon_model.proton = False
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        with self.subTest(msg='Test asymptotic behaviour --- proton electric'):
            nucleon_model.electric = True
            nucleon_model.proton = True
            at_t = nucleon_model(10000)
            at_2t = nucleon_model(20000)
            self.assertAlmostEqual(4.0, abs(at_t) / abs(at_2t), delta=0.1)

        test_cases = [
            {'t': 1.7, 'proton': True, 'electric': False,
             'expected_value': 2.000441318567961-2.743682946525417j},
            {'t': 1.7, 'proton': False, 'electric': True,
             'expected_value': 3.6578090418304723+1.8015668397588813j},
            {'t': 0.4, 'proton': True, 'electric': True,
             'expected_value': 3.928836694459375+0.839289143690434j},
            {'t': 162.42, 'proton': False, 'electric': False,
             'expected_value': -8.123212340468473e-05+6.653242386700981e-06j},
            {'t': 84.1, 'proton': True, 'electric': True,
             'expected_value': 0.0002978777900838338-3.500391752647983e-05j},
            {'t': -62.4, 'proton': False, 'electric': False,
             'expected_value': -0.0004605777184133325},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                nucleon_model.proton = case['proton']
                nucleon_model.electric = case['electric']
                actual = nucleon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
