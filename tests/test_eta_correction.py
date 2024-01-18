from unittest import TestCase
import math

from kaon_production.eta_correction import calculate_beta, dilogarithm, apply_eta_correction
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

    def test_apply_eta_correction(self):

        pass
