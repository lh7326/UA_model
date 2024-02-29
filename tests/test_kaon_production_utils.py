from unittest import TestCase
import cmath

from kaon_production.data import KaonDatapoint
from common.utils import make_partial_cross_section_for_parameters
from model_parameters import KaonParameters, KaonParametersSimplified


class TestKaonProductionUtils(TestCase):

    def setUp(self) -> None:
        self.kaon_mass = 0.493677
        self.alpha = 0.0072973525693
        self.hc_squared = 389379.3721
        self.kaon_parameters = KaonParameters(
            0.5, 0.52, 0.7, 0.8,
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
            0.5, 0.52, 0.7, 0.8,
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
            KaonDatapoint(t=0.1, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=1.812, is_charged=False, is_for_cross_section=True),
            KaonDatapoint(t=84.4301, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=87.4, is_charged=False, is_for_cross_section=True),
            KaonDatapoint(t=0.4, is_charged=True, is_for_cross_section=True),
            KaonDatapoint(t=38.4, is_charged=False, is_for_cross_section=True),
        ]
        self.expected_for_kaon_parameters = [
            6867.5008712056415,
            2.2223123606734574,
            2.6124265109483495e-05,
            0.00014068020737064747,
            287.74743207944186,
            0.0019880145131030607,
        ]

        self.expected_for_kaon_parameters_simplified = [
            6413.675586172955,
            2.3002754701557597,
            0.00012203543903522858,
            0.00015524069708229468,
            174.03966869801772,
            0.0021885490840425448,
        ]

    def assert_complex_list_close(self, actual, expected):
        for a, e in zip(actual, expected):
            self.assertTrue(cmath.isclose(a, e, abs_tol=1e-15))

    def test_make_partial_cross_section_for_parameters__kaon_parameters(self):

        with self.subTest(msg='All parameters released'):
            self.kaon_parameters.release_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters,
                charged_kaon_mass=self.kaon_mass, neutral_kaon_mass=self.kaon_mass)
            actual = f(self.ts, 0.7, 0.8, 0.1, 0.78266, 0.00868, 0.2, 1.410, 0.29,
                       0.15, 1.67, 0.315, 0.3, 1.019461, 0.004249, 0.35, 1.680, 0.150,
                       2.159, 0.137, 0.12, 0.77526, 0.1474, 0.13, 1.465, 0.4,
                       0.14, 1.720, 0.25, 2.15, 0.3)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

        with self.subTest(msg='All parameters fixed'):
            self.kaon_parameters.fix_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters,
                charged_kaon_mass = self.kaon_mass, neutral_kaon_mass = self.kaon_mass)
            actual = f(self.ts)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

        with self.subTest(msg='Some parameters free'):
            self.kaon_parameters.fix_all_parameters()
            self.kaon_parameters.release_parameters(['a_omega', 'mass_phi_prime', 'decay_rate_rho_double_prime'])
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters,
                charged_kaon_mass=self.kaon_mass, neutral_kaon_mass=self.kaon_mass)
            actual = f(self.ts, 0.1, 1.680, 0.25)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters)

    def test_make_partial_cross_section_for_parameters__kaon_parameters_simplified(self):

        with self.subTest(msg='All parameters released'):
            self.kaon_parameters_simplified.release_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters_simplified,
                charged_kaon_mass=self.kaon_mass, neutral_kaon_mass=self.kaon_mass)
            actual = f(self.ts, 0.7, 0.8, 0.2, 1.410, 0.29,
                       0.15, 1.67, 0.315, 0.3, 1.019461, 0.004249, 0.35, 1.680, 0.150,
                       2.159, 0.137, 0.13, 1.465, 0.4,
                       0.14, 1.720, 0.25, 2.15, 0.3)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)

        with self.subTest(msg='All parameters fixed'):
            self.kaon_parameters_simplified.fix_all_parameters()
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters_simplified,
                charged_kaon_mass=self.kaon_mass, neutral_kaon_mass=self.kaon_mass)
            actual = f(self.ts)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)

        with self.subTest(msg='Some parameters free'):
            self.kaon_parameters_simplified.fix_all_parameters()
            self.kaon_parameters_simplified.release_parameters(
                ['a_omega_prime', 'mass_phi_prime', 'decay_rate_rho_double_prime'])
            f = make_partial_cross_section_for_parameters(
                self.alpha, self.hc_squared, self.kaon_parameters_simplified,
                charged_kaon_mass=self.kaon_mass, neutral_kaon_mass=self.kaon_mass)
            actual = f(self.ts, 0.2, 1.680, 0.25)
            print(actual, self.expected_for_kaon_parameters_simplified)
            self.assert_complex_list_close(actual, self.expected_for_kaon_parameters_simplified)
