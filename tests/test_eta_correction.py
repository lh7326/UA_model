from unittest import TestCase
import math

from kaon_production.eta_correction import calculate_beta, dilogarithm, add_fsr_effects
# TODO: finish

class TestEtaCorrection(TestCase):

    def test_calculate_beta(self):

        arguments = [
            {'s': 2.4, 'final_particle_mass': 0.0},
            {'s': 16.0, 'final_particle_mass': 2.0},
            {'s': 21.333333333333333, 'final_particle_mass': 2.0},
            {'s': 753.2, 'final_particle_mass': 1.2},
        ]
        actual_values = [calculate_beta(**kwargs) for kwargs in arguments]
        expected_values = [
            1.0, 0.0, 0.5, 0.9961689760196,
        ]

        for kwargs, actual, expected in zip(arguments, actual_values, expected_values):
            with self.subTest(msg=str(kwargs)):
                self.assertTrue(math.isclose(actual, expected, abs_tol=1e-15))

    def test_dilogarithm(self):

        arguments = [-1, 0, 0.5, 1]
        actual_values = [dilogarithm(z) for z in arguments]
        expected_values = [
            -1 * math.pi**2 / 12,
            0.0,
            math.pi ** 2 / 12 - math.log(2)**2 / 2,
            math.pi ** 2 / 6,
        ]

        for z, actual, expected in zip(arguments, actual_values, expected_values):
            with self.subTest(msg=f'z={z}'):
                self.assertTrue(math.isclose(actual, expected, abs_tol=1e-15))

    def test_add_fsr_effects(self):

        particle_mass = 0.13957039  # charged pion
        alpha = 0.0072973525693
        test_cases = [
            {'cs': 1.0, 's': 0.08, 'expected': 1.0674789321713842},
            {'cs': 0.2, 's': 0.08, 'expected': 0.21349578643427686},
            {'cs': 1.0, 's': 0.17, 'expected': 1.0117024000513168},
            {'cs': 1.0, 's': 0.84, 'expected': 1.00765429814779},
            {'cs': 1.0, 's': 1.0, 'expected': 1.007538576779526},
            {'cs': 1.0, 's': 23.1, 'expected': 1.0069919981967508},
        ]

        for case in test_cases:
            with self.subTest(msg=f'case={case}'):
                actual = add_fsr_effects(case['cs'], case['s'], particle_mass, alpha)
                self.assertTrue(math.isclose(actual, case['expected'], abs_tol=1e-15))
