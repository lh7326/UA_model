import cmath
from unittest import TestCase
import math

from ua_model.functions import z_minus_its_reciprocal


class TestFunctions(TestCase):

    def test_z_minus_its_reciprocal(self):
        test_cases = [
            {'argument': 1, 'expected_value': 0},
            {'argument': 1j, 'expected_value': 2j},
            {'argument': -2j, 'expected_value': -2.5j},
            {'argument': 1, 'expected_value': 0},
            {'argument': (1 + 1j) / math.sqrt(2), 'expected_value': math.sqrt(2) * 1j},
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
