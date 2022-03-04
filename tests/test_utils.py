from unittest import TestCase

from ua_model.utils import compose, validate_branch_point_positions


class TestFunctionUtils(TestCase):

    def test_compose(self):

        def f(x):
            return x + 2

        def g(x):
            return 2 * x

        h = compose(f, g)
        self.assertEqual(h(0), 2)
        self.assertEqual(h(3), 8)

    def test_validate_branch_point_positions(self):
        with self.subTest(msg='valid parameters'):
            self.assertIsNone(validate_branch_point_positions(t_0=0.1, t_in=1.0))
        with self.subTest(msg='valid parameters, t_0 = 0'):
            self.assertIsNone(validate_branch_point_positions(t_0=0.0, t_in=0.1))
        with self.subTest(msg='negative t_0'):
            self.assertRaises(ValueError, validate_branch_point_positions, t_0=-0.1, t_in=1.0)
        with self.subTest(msg='t_in < t_0'):
            self.assertRaises(ValueError, validate_branch_point_positions, t_0=0.1, t_in=0.0)
