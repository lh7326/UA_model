import cmath
from unittest import TestCase
from configparser import ConfigParser

from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from ua_model.KaonUAModel import KaonUAModel


class TestScalarMesonProductionTotalCrossSection(TestCase):

    def test___call__(self):

        config = ConfigParser()
        config['constants'] = {'alpha': 1/137, 'hc_squared': 1.0}

        cross_section = ScalarMesonProductionTotalCrossSection(
            meson_mass=1.0,
            form_factor_model=lambda t: t,
            config=config,
        )

        cases = [
            {'t': 1.0, 'expected_value': -0.0002899141186372558j},
            {'t': 4.00, 'expected_value': 0.0},
            {'t': 4.72, 'expected_value': 1.5689721292965648e-05},
            {'t': 14.31 + 8.21j, 'expected_value': 0.0006291427281578755-0.00019011327189063848j},
            {'t': 862.0 - 0.87j, 'expected_value': 0.04776005175092199+4.7866214477648565e-05j},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(cmath.isclose(
                    cross_section(case['t']),
                    case['expected_value'],
                    abs_tol=1e-15,
                ))

    def test___call___kaon_ua_model(self):
        m_pion = 0.13957039

        kaon_model = KaonUAModel(
            charged_variant=True,
            t_0_isoscalar=(9 * m_pion ** 2),
            t_0_isovector=(4 * m_pion ** 2),
            t_in_isoscalar=1.35,
            t_in_isovector=0.59,
            a_omega=0.004,
            a_omega_prime=0.27,
            a_omega_double_prime=0.0,
            a_phi=-0.01,
            a_phi_prime=0.0,
            a_rho=0.24,
            a_rho_prime=0.1,
            a_rho_double_prime=-0.1,
            mass_omega=0.78266,
            decay_rate_omega=0.00868,
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
            mass_rho=0.77525,
            decay_rate_rho=0.1474,
            mass_rho_prime=1.465,
            decay_rate_rho_prime=0.4,
            mass_rho_double_prime=1.570,
            decay_rate_rho_double_prime=0.144,
            mass_rho_triple_prime=1.720,
            decay_rate_rho_triple_prime=0.25,
        )

        config = ConfigParser()
        config['constants'] = {
            'alpha': 0.0072973525693,
            'hc_squared': 389379.3721,  # in GeV^2 * nb
        }

        cross_section = ScalarMesonProductionTotalCrossSection(
            meson_mass=0.493677,
            form_factor_model=kaon_model,
            config=config,
        )

        test_cases = [
            {'t': 1.1230, 'expected_value': 1.422899583299445},
            {'t': 1.1230 + 0.73200j, 'expected_value': 7.322169806015914+5.145404160322535j},
            {'t': 0.73200j, 'expected_value': 43.88590980666062-8.017152698541237j},
            {'t': -0.73200j, 'expected_value': 43.88590980666064+8.01715269854124j},
            {'t': 1.1230 - 0.73200j, 'expected_value': 7.32216980601591-5.145404160322531j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = cross_section(case['t'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1e-15))
