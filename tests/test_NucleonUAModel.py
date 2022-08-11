from unittest import TestCase
import cmath

from ua_model.NucleonUAModel import NucleonUAModel


class TestNucleonUAModel(TestCase):

    def test___call__(self):
        nucleon_model = NucleonUAModel(
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
            a_pauli_omega_prime=0.09,
            a_pauli_phi=-0.42,
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

        with self.subTest(msg='Test constraints --- dirac isoscalar'):
            sum_alphas = (nucleon_model.a_dirac_omega + nucleon_model.a_dirac_omega_prime +
                          nucleon_model.a_dirac_omega_double_prime + nucleon_model.a_dirac_phi +
                          nucleon_model.a_dirac_phi_prime + nucleon_model.a_dirac_phi_double_prime)
            sum_alphas_times_m_sq = (
                    nucleon_model.a_dirac_omega * nucleon_model.mass_omega**2 +
                    nucleon_model.a_dirac_omega_prime * nucleon_model.mass_omega_prime**2 +
                    nucleon_model.a_dirac_omega_double_prime * nucleon_model.mass_omega_double_prime**2 +
                    nucleon_model.a_dirac_phi * nucleon_model.mass_phi**2 +
                    nucleon_model.a_dirac_phi_prime * nucleon_model.mass_phi_prime**2 +
                    nucleon_model.a_dirac_phi_double_prime * nucleon_model.mass_phi_double_prime**2)
            self.assertAlmostEqual(sum_alphas, 0.5)
            self.assertAlmostEqual(sum_alphas_times_m_sq, 0.0)

        with self.subTest(msg='Test constraints --- dirac isovector'):
            sum_alphas = (nucleon_model.a_dirac_rho + nucleon_model.a_dirac_rho_prime +
                          nucleon_model.a_dirac_rho_double_prime)
            sum_alphas_times_m_sq = (
                    nucleon_model.a_dirac_rho * nucleon_model.mass_rho ** 2 +
                    nucleon_model.a_dirac_rho_prime * nucleon_model.mass_rho_prime ** 2 +
                    nucleon_model.a_dirac_rho_double_prime * nucleon_model.mass_rho_double_prime ** 2)
            self.assertAlmostEqual(sum_alphas, 0.5)
            self.assertAlmostEqual(sum_alphas_times_m_sq, 0.0)

        with self.subTest(msg='Test constraints --- pauli isoscalar'):
            sum_alphas = (nucleon_model.a_pauli_omega + nucleon_model.a_pauli_omega_prime +
                          nucleon_model.a_pauli_omega_double_prime + nucleon_model.a_pauli_phi +
                          nucleon_model.a_pauli_phi_prime + nucleon_model.a_pauli_phi_double_prime)
            sum_alphas_times_m_sq = (
                    nucleon_model.a_pauli_omega * nucleon_model.mass_omega**2 +
                    nucleon_model.a_pauli_omega_prime * nucleon_model.mass_omega_prime**2 +
                    nucleon_model.a_pauli_omega_double_prime * nucleon_model.mass_omega_double_prime**2 +
                    nucleon_model.a_pauli_phi * nucleon_model.mass_phi**2 +
                    nucleon_model.a_pauli_phi_prime * nucleon_model.mass_phi_prime**2 +
                    nucleon_model.a_pauli_phi_double_prime * nucleon_model.mass_phi_double_prime**2)
            sum_alphas_times_m_4 = (
                    nucleon_model.a_pauli_omega * nucleon_model.mass_omega ** 4 +
                    nucleon_model.a_pauli_omega_prime * nucleon_model.mass_omega_prime ** 4 +
                    nucleon_model.a_pauli_omega_double_prime * nucleon_model.mass_omega_double_prime ** 4 +
                    nucleon_model.a_pauli_phi * nucleon_model.mass_phi ** 4 +
                    nucleon_model.a_pauli_phi_prime * nucleon_model.mass_phi_prime ** 4 +
                    nucleon_model.a_pauli_phi_double_prime * nucleon_model.mass_phi_double_prime ** 4)
            self.assertAlmostEqual(sum_alphas, 0.5)
            self.assertAlmostEqual(sum_alphas_times_m_sq, 0.0)
            self.assertAlmostEqual(sum_alphas_times_m_4, 0.0)

        with self.subTest(msg='Test constraints --- pauli isovector'):
            sum_alphas = (nucleon_model.a_pauli_rho + nucleon_model.a_pauli_rho_prime +
                          nucleon_model.a_pauli_rho_double_prime)
            sum_alphas_times_m_sq = (
                    nucleon_model.a_pauli_rho * nucleon_model.mass_rho ** 2 +
                    nucleon_model.a_pauli_rho_prime * nucleon_model.mass_rho_prime ** 2 +
                    nucleon_model.a_pauli_rho_double_prime * nucleon_model.mass_rho_double_prime ** 2)
            sum_alphas_times_m_4 = (
                    nucleon_model.a_pauli_rho * nucleon_model.mass_rho ** 4 +
                    nucleon_model.a_pauli_rho_prime * nucleon_model.mass_rho_prime ** 4 +
                    nucleon_model.a_pauli_rho_double_prime * nucleon_model.mass_rho_double_prime ** 4)
            self.assertAlmostEqual(sum_alphas, -1.2)
            self.assertAlmostEqual(sum_alphas_times_m_sq, 0.0)
            self.assertAlmostEqual(sum_alphas_times_m_4, 0.0)

        test_cases = [
            {'t': 1.7, 'proton': True, 'electric': False,
             'expected_value': 4.36717975188465+0.09514920657417147j},
            {'t': 1.7, 'proton': False, 'electric': True,
             'expected_value': 4.290219270325238-0.053025055478032886j},
            {'t': 0.4+1.2j, 'proton': True, 'electric': True,
             'expected_value': 0.8062489164241646+0.20344542716556585j},
            {'t': 162.42-0.647j, 'proton': False, 'electric': False,
             'expected_value': -0.014561505124673899-0.0002236384688640401j},
            {'t': 84.1-9124.1j, 'proton': True, 'electric': True,
             'expected_value': 0.26563707978916934+0.0004447372971235849j},
            {'t': 62.4j, 'proton': False, 'electric': False,
             'expected_value': 0.019071884752262992+0.01373477520811673j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                nucleon_model.proton = case['proton']
                nucleon_model.electric = case['electric']
                actual = nucleon_model(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
