from unittest import TestCase
import cmath

from ua_model.MapFromTtoW import MapFromTtoW
from ua_model.MapFromWtoT import MapFromWtoT


class TestCoordinateMapsConsistency(TestCase):

    def test_TtoWtoT__real_t_physical_sheet(self):
        """
        Test that MapFromWtoT is the inverse of MapFromTtoW. We only use the physical sheet
        real t values.

        """
        t_0 = 0.48
        t_in = 17.3
        t_to_w = MapFromTtoW(t_0, t_in)
        w_to_t = MapFromWtoT(t_0, t_in)

        for t in [-1.2, 0.0, 0.3, 1.2, 78.4, 0.001712431, -32.16, -92641.2451, 81.20021]:
            with self.subTest(t=t):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w(t)),
                    abs_tol=1.0e-14,
                ))

    def test_WtoTtoW__real_t_physical_sheet(self):
        """
        Test that for W corresponding to a real physical sheet value of t the composition
        of the maps WtoT and TtoW is the identity.

        Real physical sheet values of t are mapped onto the union of [-1, 0], [0, 1j], upper
        left fourth of the unit circle.

        """
        t_0 = 71.251
        t_in = 72.004
        w_to_t = MapFromWtoT(t_0, t_in)
        t_to_w = MapFromTtoW(t_0, t_in)

        for w in [
            -0.999, -0.741, -0.371, -0.108, -0.031,
            0.003j, 0.41j, 0.54j, 0.82j, 0.9199412j,
            cmath.exp(1j * cmath.pi * 0.53), cmath.exp(1j * cmath.pi * 0.74),
        ]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w(w_to_t(w).real),
                    abs_tol=1.0e-14,
                ))

    def test_TtoWtoT(self):
        """
        Test that MapFromWtoT is the inverse of MapFromTtoW.
        We stay away from the cuts and consider various sheets in the Riemann surface of t.

        """
        t_0 = 0.48
        t_in = 17.3
        t_to_w = MapFromTtoW(t_0, t_in)
        w_to_t = MapFromWtoT(t_0, t_in)

        for t in [-75.5214, -1.2, 0.0, 0.037851, 0.3, 0.4731, 1j, -0.1j, 816.6412 + 76.1j,
                  -0.2+14.2j, -72.413-0.0081j, 42.7-871264.982j]:
            with self.subTest(msg=f't={t}, sheet 1'):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w.map_from_sheet(t, 1)),
                    abs_tol=1.0e-14,
                ))

        for t in [-75.5214, -1.2, 0.0, 0.037851, 0.3, 0.4731, 1j, -0.1j, 816.6412 + 76.1j,
                  -0.2 + 14.2j, -72.413 - 0.0081j, 42.7 - 871264.982j]:
            with self.subTest(msg=f't={t}, sheet 2'):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w.map_from_sheet(t, 2)),
                    abs_tol=1.0e-14,
                ))

        for t in [-75.5214, -1.2, 0.0, 0.037851, 0.3, 0.4731, 1j, -0.1j, 816.6412 + 76.1j,
                  -0.2 + 14.2j, -72.413 - 0.0081j, 42.7 - 871264.982j]:
            with self.subTest(msg=f't={t}, sheet 3'):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w.map_from_sheet(t, 3)),
                    abs_tol=1.0e-14,
                ))

        for t in [-75.5214, -1.2, 0.0, 0.037851, 0.3, 0.4731, 1j, -0.1j, 816.6412 + 76.1j,
                  -0.2 + 14.2j, -72.413 - 0.0081j, 42.7 - 871264.982j]:
            with self.subTest(msg=f't={t}, sheet 4'):
                self.assertTrue(cmath.isclose(
                    t,
                    w_to_t(t_to_w.map_from_sheet(t, 4)),
                    abs_tol=1.0e-14,
                ))

    def test_WtoTtoW(self):
        """
        Test that MapFromTtoW is the inverse of MapFromWtoT.
        We stay away from the cuts and consider various sheets in the Riemann surface of t.

        """
        t_0 = 71.251
        t_in = 72.004
        w_to_t = MapFromWtoT(t_0, t_in)
        t_to_w = MapFromTtoW(t_0, t_in)

        for w in [-0.9987, -0.741, -0.4382, -0.2315, -0.003931,
                  -0.05+0.82j, -0.491-0.08j, -0.251+0.38j,
                  -0.572-0.687j, -0.0002-0.92j]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w.map_from_sheet(w_to_t(w), 1),
                    abs_tol=1.0e-14,
                ))

        for w in [0.7982, 0.941, 0.4321, 0.2154, 0.002931,
                  0.04+0.85j, 0.491-0.07j, 0.251+0.38j,
                  0.572-0.629j, 0.0002+0.92j]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w.map_from_sheet(w_to_t(w), 2),
                    abs_tol=1.0e-14,
                ))

        for w in [-1891.653, -723.7, -46.1, -1.74,
                  -0.04+1.85j, -0.491-2.35j, -0.251+1834.861j,
                  -9123.572-0.629j, -7861.2314+0.92j, -0.931+0.712j, -1.4+1.02j]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w.map_from_sheet(w_to_t(w), 3),
                    rel_tol=1.0e-7,
                ))

        for w in [1813.653, 453.2, 72.3, 1.42, 1.023,
                  0.04+1.85j, 0.491-2.35j, 0.251+1834.861j,
                  1723.572-0.629j, 361.714+0.92j, 0.931+0.712j, 0.314+1.02j]:
            with self.subTest(w=w):
                self.assertTrue(cmath.isclose(
                    w,
                    t_to_w.map_from_sheet(w_to_t(w), 4),
                    rel_tol=1.0e-7,
                ))
