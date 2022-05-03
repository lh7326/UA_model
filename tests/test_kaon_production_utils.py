from unittest import TestCase
import cmath

from kaon_production.Task import Datapoint
from kaon_production.utils import make_partial_cross_section_for_parameters
from model_parameters import KaonParameters, KaonParametersSimplified


class TestKaonProductionUtils(TestCase):

    def setUp(self) -> None:
        self.kaon_mass = 0.493677
        self.alpha = 0.0072973525693
        self.hc_squared = 389379.3721
        self.kaon_parameters = KaonParameters(
            0.5, 0.6, 0.7, 0.8,
            0.1, 0.78266, 0.00868,
            0.2, 1.410, 0.29,
            0.15, 1.67, 0.315,
            0.3, 1.019461, 0.004249,
            0.35, 1.680, 0.150,
            2.159, 0.137,
            0.12, 0.77526, 0.1474,
            0.13, 1.465, 0.4,
            0.14, 1.720, 0.25,
            2.15, 0.3,
        )
        self.kaon_parameters_simplified = KaonParametersSimplified(
            0.5, 0.6, 0.7, 0.8,
            0.2, 1.410, 0.29,
            0.15, 1.67, 0.315,
            0.3, 1.019461, 0.004249,
            0.35, 1.680, 0.150,
            2.159, 0.137,
            0.13, 1.465, 0.4,
            0.14, 1.720, 0.25,
            2.15, 0.3,
        )
        self.ts = [
            Datapoint(t=0.1, is_charged=True),
            Datapoint(t=1.812, is_charged=False),
            Datapoint(t=84.4301, is_charged=True),
            Datapoint(t=87.4j, is_charged=False),
            Datapoint(t=0.4 - 0.02j, is_charged=True),
            Datapoint(t=38.4 + 7.93j, is_charged=False),
        ]
        self.expected_for_kaon_parameters = [
            -1.2547114651583273e-12-6830.331085566366j,
            1.860659317681214,
            6.1503729934415144e-06,
            1.908862950293916e-06-0.00011408449615025597j,
            -48.070421296077555+268.647889447054j,
            0.0016394233088188996-0.0003253646031991748j,
        ]

        self.expected_for_kaon_parameters_simplified = [
            -1.164379836180649e-12-6338.588600900197j,
            1.8799848490673723,
            6.200206478745468e-05,
            2.0612135774974183e-06-0.0001231898352946984j,
            -28.56465672307159+159.63735150494622j,
            0.0017544204413164524-0.00034818726052192876j,
        ]

    @staticmethod
    def assert_complex_list_close(actual, expected):
        for a, e in zip(actual, expected):
            cmath.isclose(a, e, abs_tol=1e-15)

    def test_make_partial_cross_section_for_parameters__kaon_parameters(self):

        with self.subTest(msg='All parameters released'):
            self.kaon_parameters.release_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters)
            actual = f(self.ts, 0.7, 0.8, 0.1, 0.78266, 0.00868, 0.2, 1.410, 0.29,
                       0.15, 1.67, 0.315, 0.3, 1.019461, 0.004249, 0.35, 1.680, 0.150,
                       2.159, 0.137, 0.12, 0.77526, 0.1474, 0.13, 1.465, 0.4,
                       0.14, 1.720, 0.25, 2.15, 0.3)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

        with self.subTest(msg='All parameters fixed'):
            self.kaon_parameters.fix_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters)
            actual = f(self.ts)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

        with self.subTest(msg='Some parameters free'):
            self.kaon_parameters.fix_all_parameters()
            self.kaon_parameters.release_parameters(['a_omega', 'mass_phi_prime', 'decay_rate_rho_double_prime'])
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters)
            actual = f(self.ts, 0.1, 1.680, 0.25)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

    def test_make_partial_cross_section_for_parameters__kaon_parameters_simplified(self):

        with self.subTest(msg='All parameters released'):
            self.kaon_parameters_simplified.release_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters_simplified)
            actual = f(self.ts, 0.7, 0.8, 0.2, 1.410, 0.29,
                       0.15, 1.67, 0.315, 0.3, 1.019461, 0.004249, 0.35, 1.680, 0.150,
                       2.159, 0.137, 0.13, 1.465, 0.4,
                       0.14, 1.720, 0.25, 2.15, 0.3)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)

        with self.subTest(msg='All parameters fixed'):
            self.kaon_parameters_simplified.fix_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters_simplified)
            actual = f(self.ts)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)

        with self.subTest(msg='Some parameters free'):
            self.kaon_parameters_simplified.fix_all_parameters()
            self.kaon_parameters_simplified.release_parameters(
                ['a_omega_prime', 'mass_phi_prime', 'decay_rate_rho_double_prime'])
            f = make_partial_cross_section_for_parameters(
                self.kaon_mass, self.alpha, self.hc_squared, self.kaon_parameters_simplified)
            actual = f(self.ts, 0.1, 1.680, 0.25)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)
