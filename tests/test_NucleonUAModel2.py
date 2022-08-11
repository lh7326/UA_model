from unittest import TestCase
import cmath

from ua_model.NucleonUAModel2 import NucleonUAModel2


class TestNucleonUAModel2(TestCase):

    def test___call__(self):
        nucleon_model = NucleonUAModel2(
            proton=True,
            electric=True,
            mass=1.0,
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
             'expected_value': 9.264041524614782+0.07288173635849036j},
            {'t': 1.7, 'proton': False, 'electric': True,
             'expected_value': 6.36801456730032-0.05590379378968173j},
            {'t': 0.4+1.2j, 'proton': True, 'electric': True,
             'expected_value': 0.7782203362803921+0.1525683395543666j},
            {'t': 162.42-0.647j, 'proton': False, 'electric': False,
             'expected_value': -0.002759958992972902-2.598171690422949e-05j},
            {'t': 84.1-9124.1j, 'proton': True, 'electric': True,
             'expected_value': -5.1303718120553975e-06+7.795611633972505e-08j},
            {'t': 62.4j, 'proton': False, 'electric': False,
             'expected_value': 0.009671246342365719-0.009930048457120286j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                nucleon_model.proton = case['proton']
                nucleon_model.electric = case['electric']
                actual = nucleon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
