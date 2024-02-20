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
            {'t': 1.0, 'expected_W': -0.86602540378+0.5j},
            {'t': 12.7, 'expected_W': -0.9901085600225469+0.14030338331657843j}
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(
                    cmath.isclose(f(case['t']), case['expected_W'], abs_tol=1e-15)
                )

    def test_map_from_sheet(self):

        map_from_t_to_w = MapFromTtoW(t_0=0.0, t_in=0.25)

        test_cases = [
            {'t': 0.031069896194 + 0.035343944637j, 'sheet': 1, 'expected_W': -0.1 + 0.2j},
            {'t': 0.031069896194 + 0.035343944637j, 'sheet': 2, 'expected_W': +0.1 - 0.2j},
            {'t': 0.031069896194 + 0.035343944637j, 'sheet': 3, 'expected_W': 1/(-0.1+0.2j)},
            {'t': 0.031069896194 + 0.035343944637j, 'sheet': 4, 'expected_W': 1/(0.1-0.2j)},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(
                    cmath.isclose(map_from_t_to_w.map_from_sheet(case['t'], case['sheet']),
                                  case['expected_W'], abs_tol=1e-15)
                )
