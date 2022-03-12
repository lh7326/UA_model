from unittest import TestCase
import cmath

from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB


class TestUAComponentVariantB(TestCase):

    def test___call__(self):
        component = UAComponentVariantB(w_n=2, w_meson=3 + 1j)

        test_cases = [
            {'w': 0.0, 'expected_value': 0.057777777777777775},
            {'w': 1.0, 'expected_value': 0.0},
            {'w': 1j, 'expected_value': 0.19753086419753085},
            {'w': 2.1 - 4.3j, 'expected_value': 3.5515073618039743+1.5943390598942047j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = component(case['w'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected))
