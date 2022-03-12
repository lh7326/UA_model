from unittest import TestCase
import cmath

from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA


class TestUAComponentVariantA(TestCase):

    def test___call__(self):
        component = UAComponentVariantA(w_n=2, w_meson=3 + 1j)

        test_cases = [
            {'w': 0.0, 'expected_value': 0.6444444444444444},
            {'w': 1.0, 'expected_value': 0.0},
            {'w': 1j, 'expected_value': -0.22032288698955366},
            {'w': 2.1 - 4.3j, 'expected_value': 0.15935491958039452+0.8787611797211575j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = component(case['w'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected))
