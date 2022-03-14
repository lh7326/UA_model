from unittest import TestCase
import cmath

from ua_model.MapFromTtoW import MapFromTtoW
from ua_model.MapFromWtoT import MapFromWtoT


class TestCoordinateMapsConsistency(TestCase):

    def test_TtoWtoT(self):
        """
        Test that MapFromWtoT is the inverse of MapFromTtoW.

        """
        t_0 = 0.48
        t_in = 17.3
        t_to_w = MapFromTtoW(t_0, t_in)
        w_to_t = MapFromWtoT(t_0, t_in)

        for t in [-1.2, 0.0, 0.3, 1.2, 78.4, 0.001712431, 1j, -0.1j, 816.6412 + 76.1j,
                  -0.2+14.2j, -72.413-0.0081j, 42.7 - 871264.982j]:
            with self.subTest(t=t):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w(t)),
                    abs_tol=1.0e-15,
                ))

    def test_WtoTtoW(self):
        """
        Test that for W from the left unit semi-disk (centered at the origin) the composition
        of the maps WtoT and TtoW is the identity.

        """
        t_0 = 71.251
        t_in = 72.004
        w_to_t = MapFromWtoT(t_0, t_in)
        t_to_w = MapFromTtoW(t_0, t_in)

        for w in [-0.108, -0.371, -0.741, -0.999, -0.05 + 0.98j, -0.491 - 0.08j,
                  -0.251 + 0.38j, -0.572 - 0.687j]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w(w_to_t(w)),
                    abs_tol=1.0e-15,
                ))
