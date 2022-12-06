import cmath
from unittest import TestCase
from configparser import ConfigParser

from cross_section.NucleonPairToElectronPositronTotalCrossSection import NucleonPairToElectronPositronTotalCrossSection
from ua_model.NucleonUAModel import NucleonUAModel


class TestScalarMesonProductionTotalCrossSection(TestCase):

    def test___call__(self):

        config = ConfigParser()
        config['constants'] = {'alpha': 1/137, 'hc_squared': 1.0}

        cross_section = NucleonPairToElectronPositronTotalCrossSection(
            nucleon_mass=1.0,
            form_factor_model=lambda t: t,
            config=config,
        )

        cases = [
            {'t': 4.1, 'expected_value': 0.00021261082938175543},
            {'t': 4.21, 'expected_value': 0.0003095335115185389},
            {'t': 14.31 + 8.21j, 'expected_value': 0.0029757326393812914-0.0019301384158627749j},
            {'t': 862.0 - 0.87j, 'expected_value': 0.1923761490892198+0.0001946109906016082j},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(cmath.isclose(
                    cross_section(case['t']),
                    case['expected_value'],
                    abs_tol=1e-15,
                ))

    def test___call___nucleon_ua_model(self):
        m_proton = 0.938272081

        nucleon_model = NucleonUAModel(
            proton=True,
            electric=True,
            nucleon_mass=m_proton,
            magnetic_moment_proton=2.792847351,
            magnetic_moment_neutron=-1.91304273,
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

        config = ConfigParser()
        config['constants'] = {
            'alpha': 0.0072973525693,
            'hc_squared': 389379.3721,  # in GeV^2 * nb
        }

        cross_section = NucleonPairToElectronPositronTotalCrossSection(
            nucleon_mass=m_proton,
            form_factor_model=nucleon_model,
            config=config,
        )

        test_cases = [
            {'t': 4.0, 'expected_value': 4.037438660548728},
            {'t': 4.1230 + 0.73200j, 'expected_value': 1.074340393409335-0.3040419016151704j},
            {'t': 3.73200j, 'expected_value': -0.019159837461526916-0.5369667339262477j},
            {'t': -7.73200j, 'expected_value': -0.0007610111434823791+0.0474064128632346j},
            {'t': 12.1230 - 0.73200j, 'expected_value': 0.007284922223154086+0.0004451300075981765j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = cross_section(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))
