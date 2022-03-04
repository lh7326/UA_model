import cmath
from unittest import TestCase

from ua_model.MapFromWtoT import MapFromWtoT


class TestMapFromWtoT(TestCase):

    def test_validation(self):

        self.assertIsInstance(MapFromWtoT(t_0=1.0, t_in=2.0), MapFromWtoT)  # OK
        self.assertRaises(ValueError, MapFromWtoT, t_0=-0.1, t_in=1.0)  # a negative parameter
        self.assertRaises(ValueError, MapFromWtoT, t_0=0.1, t_in=0.0)  # t_in < t_0

    def test___call__(self):

        f = MapFromWtoT(t_0=0.0, t_in=0.25)

        test_cases = [
            {'W': 2.0, 'expected_t': -4.0 / 9.0},
            {'W': 0.5j, 'expected_t': 4.0 / 25.0},
            {'W': -2j, 'expected_t': 4.0 / 25.0},
            {'W': -0.1 + 0.2j, 'expected_t': 0.031069896194 + 0.035343944637j},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(
                    cmath.isclose(f(case['W']), case['expected_t'])
                )
