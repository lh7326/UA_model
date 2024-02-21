from unittest import TestCase
import cmath
from ua_model.MapFromTtoW import MapFromTtoW

from ua_model.ua_components.UAComponentVariantA import UAComponentVariantA


class TestUAComponentVariantA(TestCase):

    def setUp(self):
        self.map_from_t_to_w = MapFromTtoW(t_0=1.0, t_in=17)

    def test___call__(self):
        component = UAComponentVariantA(meson_mass=4.0, meson_decay_rate=2.0, map_from_t_to_w=self.map_from_t_to_w)

        test_cases = [
            {'w': -0.12310562561766053, 'expected_value': 1.0},
            {'w': 0.0, 'expected_value': 1.388250084407719},
            {'w': 0.5, 'expected_value': 1.2035334340523391},
            {'w': 1, 'expected_value': 0.0},
            {'w': -1, 'expected_value': 0.0},
            {'w': 0.5j, 'expected_value': 0.15104669065464943+2.527844870718135j},
            {'w': 0.95j, 'expected_value': -2.7000968058743635+0.31047939094436844j},
            {'w': cmath.exp(1j * cmath.pi * 0.6), 'expected_value': -1.3090976397159548+2.2e-16j},
            {'w': cmath.exp(1j * cmath.pi * 0.9), 'expected_value': -0.05302262750821873+1e-17j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = component(case['w'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected))
