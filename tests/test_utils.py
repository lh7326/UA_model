from unittest import TestCase

from ua_model.utils import validate_branch_point_positions


class TestFunctionUtils(TestCase):

    def test_validate_branch_point_positions(self):
        with self.subTest(msg='valid parameters'):
            self.assertIsNone(validate_branch_point_positions(t_0=0.1, t_in=1.0))
        with self.subTest(msg='valid parameters, t_0 = 0'):
            self.assertIsNone(validate_branch_point_positions(t_0=0.0, t_in=0.1))
        with self.subTest(msg='negative t_0'):
            self.assertRaises(ValueError, validate_branch_point_positions, t_0=-0.1, t_in=1.0)
        with self.subTest(msg='t_in < t_0'):
            self.assertRaises(ValueError, validate_branch_point_positions, t_0=0.1, t_in=0.0)
