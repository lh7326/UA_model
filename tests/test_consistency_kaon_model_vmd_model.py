from unittest import TestCase
import cmath

from ua_model.VMDModel import VMDModel
from ua_model.KaonUAModel import KaonUAModel


class TestConsistencyKaonModelVMDModel(TestCase):
    """
    For zero decay widths, the corresponding U&A model and VMD model should give same values.

    """

    def test_vmd_model_ua_model_consistency(self):

        kaon_model = KaonUAModel(
            charged_variant=True,
            t_0_isoscalar=0.03,
            t_in_isoscalar=1.72,
            t_0_isovector=1.37,
            t_in_isovector=847.2,
            a_omega=0.13,
            a_omega_prime=0.02,
            a_omega_double_prime=0.01,
            a_phi=-0.05,
            a_phi_prime=0.24,
            a_rho=-0.08,
            a_rho_prime=0.27,
            a_rho_double_prime=-0.14,
            mass_omega=0.23,
            decay_rate_omega=0.0,
            mass_omega_prime=0.47,
            decay_rate_omega_prime=0.0,
            mass_omega_double_prime=0.89,
            decay_rate_omega_double_prime=0.0,
            mass_phi=1.91,
            decay_rate_phi=0.0,
            mass_phi_prime=4.78,
            decay_rate_phi_prime=0.0,
            mass_phi_double_prime=9.57,
            decay_rate_phi_double_prime=0.0,
            mass_rho=1.42,
            decay_rate_rho=0.0,
            mass_rho_prime=7.15,
            decay_rate_rho_prime=0.0,
            mass_rho_double_prime=28.1,
            decay_rate_rho_double_prime=0.0,
            mass_rho_triple_prime=592.2,
            decay_rate_rho_triple_prime=0.0,
        )

        vmd_model = VMDModel.create([
            {'coefficient': 0.13, 'mass': 0.23},
            {'coefficient': 0.02, 'mass': 0.47},
            {'coefficient': 0.01, 'mass': 0.89},
            {'coefficient': -0.05, 'mass': 1.91},
            {'coefficient': 0.24, 'mass': 4.78},
            {'coefficient': 0.15, 'mass': 9.57},
            {'coefficient': -0.08, 'mass': 1.42},
            {'coefficient': 0.27, 'mass': 7.15},
            {'coefficient': -0.14, 'mass': 28.1},
            {'coefficient': 0.45, 'mass': 592.2},
        ])

        for t in [0.000, 0.00274, 0.871, 1.742, 7.81702, 9182.3891, 91884124214.21,
                  0.7451 + 0.024j, 0.087912 - 0.912421j, 97.21 - 9985982.24j, 87672819487.42j]:
            with self.subTest(t=t):
                self.assertTrue(cmath.isclose(
                    kaon_model(t),
                    vmd_model(t),
                    abs_tol=1.0e-15,
                ))
