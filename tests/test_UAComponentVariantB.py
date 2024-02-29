from unittest import TestCase
import cmath
from ua_model.MapFromTtoW import MapFromTtoW

from ua_model.ua_components.UAComponentVariantB import UAComponentVariantB


class TestUAComponentVariantB(TestCase):

    def setUp(self):
        self.map_from_t_to_w = MapFromTtoW(t_0=1.0, t_in=17)

    def test___call__(self):
        component = UAComponentVariantB(meson_mass=5.0, meson_decay_rate=2.0, map_from_t_to_w=self.map_from_t_to_w)

        test_cases = [
            {'w': -0.12310562561766053, 'expected_value': 1.0},
            {'w': 0.0, 'expected_value': 1.0339846232952001},
            {'w': -0.5, 'expected_value': 0.5438770272319838},
            {'w': 1, 'expected_value': 0.0},
            {'w': -1, 'expected_value': 0.0},
            {'w': 0.2j, 'expected_value': 1.1259986814075789},
            {'w': 0.95j, 'expected_value': 3.2928963787161694},
            {'w': cmath.exp(1j * cmath.pi * 0.6), 'expected_value': 3.4874668977409833+1.3906837226933497j},
            {'w': cmath.exp(1j * cmath.pi * 0.8), 'expected_value': -1.1962184257000497+1.1738657942053017j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                actual = component(case['w'])
                expected = case['expected_value']
                self.assertTrue(cmath.isclose(actual, expected))
