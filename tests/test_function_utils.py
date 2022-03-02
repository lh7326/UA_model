from unittest import TestCase

from ua_model.function_utils import compose


class TestFunctionUtils(TestCase):

    def test_compose(self):

        def f(x):
            return x + 2

        def g(x):
            return 2 * x

        h = compose(f, g)
        self.assertEqual(h(0), 2)
        self.assertEqual(h(3), 8)
