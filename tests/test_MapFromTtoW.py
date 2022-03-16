import cmath
from unittest import TestCase

from ua_model.MapFromTtoW import MapFromTtoW


class TestMapFromTtoW(TestCase):

    def test_validation(self):

        self.assertIsInstance(MapFromTtoW(t_0=1.0, t_in=2.0), MapFromTtoW)  # OK
        self.assertRaises(ValueError, MapFromTtoW, t_0=-0.1, t_in=1.0)  # a negative parameter
        self.assertRaises(ValueError, MapFromTtoW, t_0=0.1, t_in=0.0)  # t_in < t_0

    def test___call__(self):

        f = MapFromTtoW(t_0=0.0, t_in=0.25)

        test_cases = [
            {'t': -4.0 / 9.0, 'expected_W': -0.5},
            {'t': 4.0 / 25.0, 'expected_W': 0.5j},
            {'t': 1.0, 'expected_W': (-0.86602540378 + 0.5j)},
            {'t': 0.031069896194 + 0.035343944637j, 'expected_W': -0.1 + 0.2j},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(
                    cmath.isclose(f(case['t']), case['expected_W'])
                )
