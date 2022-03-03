import cmath
from unittest import TestCase
import math

from ua_model.functions import z_minus_its_reciprocal, square_root


class TestFunctions(TestCase):

    def test_z_minus_its_reciprocal(self):

        sqrt_2 = 1.4142135623730951

        test_cases = [
            {'argument': 1, 'expected_value': 0},
            {'argument': 1j, 'expected_value': 2j},
            {'argument': -2j, 'expected_value': -2.5j},
            {'argument': 1, 'expected_value': 0},
            {'argument': (1 + 1j) / sqrt_2, 'expected_value': sqrt_2 * 1j},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(
                    cmath.isclose(
                        z_minus_its_reciprocal(case['argument']),
                        case['expected_value']),
                )

    def test_z_minus_its_reciprocal__symmetry(self):
        """Test that f(z) = f(-1/z)"""

        for z in [0.5, -1j, -2.3 + 4j, 6712 - 76j, -43 - 1j, 0.7612 + 1j, 0.0004 + 0.001j]:
            with self.subTest(z=z):
                actual = z_minus_its_reciprocal(-1/z)
                expected = z_minus_its_reciprocal(z)
                self.assertTrue(cmath.isclose(actual, expected))

    def test_square_root(self):
        """Test our 'custom' branch of the square root"""

        sqrt_2 = 1.4142135623730951
        sqrt_534 = 23.108440016582687

        test_cases = [
            {'argument': 1, 'expected_value': 1},
            {'argument': -1, 'expected_value': 1j},
            {'argument': 1j, 'expected_value': (1 + 1j) / sqrt_2},
            {'argument': -1j, 'expected_value': (-1 + 1j) / sqrt_2},
            {'argument': 4, 'expected_value': 2},
            {'argument': 534, 'expected_value': sqrt_534},
            {'argument': -3 - 4j, 'expected_value': -1 + 2j},
            {'argument': 10000 - 0.000000001j, 'expected_value': -100},
            {'argument': -9j, 'expected_value': (-3 + 3j) / sqrt_2},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                actual = square_root(case['argument'])
                self.assertTrue(cmath.isclose(actual, case['expected_value']))
