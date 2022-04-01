from unittest import TestCase
import numpy as np

from kaon_production.ModelParameters import ModelParameters, Parameter


class TestModelParameters(TestCase):

    def setUp(self):
        self.parameters = ModelParameters(
            0.5, 0.6, 0.7, 0.8,
            0.1, 0.78266, 0.00868,
            0.2, 1.410, 0.29,
            0.15, 1.67, 0.315,
            0.3, 1.019461, 0.004249,
            0.35, 1.680, 0.150,
            2.159, 0.137,
            0.12, 0.77526, 0.1474,
            0.13, 1.465, 0.4,
            0.14, 1.720, 0.25,
            2.15, 0.3,
        )

    def test___getitem__(self):
        with self.subTest():
            par1 = self.parameters['a_rho_double_prime']
            self.assertEqual(par1.value, 0.14)
        with self.subTest():
            par2 = self.parameters['mass_phi_prime']
            self.assertEqual(par2.value, 1.680)
        with self.subTest():
            par3 = self.parameters['t_in_isovector']
            self.assertEqual(par3.value, 0.8)
        with self.subTest():
            with self.assertRaises(KeyError):
                _ = self.parameters['non-existent-key']

    def test___setitem__(self):
        with self.subTest(msg='Testing type checking'):
            with self.assertRaises(TypeError):
                self.parameters['a_omega'] = 2.0

        with self.subTest(msg='Testing parameter name checking'):
            with self.assertRaises(ValueError):
                self.parameters['a_omega'] = Parameter(name='new name', value=2.0, is_fixed=True)

        with self.subTest(msg='Testing a successful call'):
            self.parameters['a_omega'] = Parameter(name='a_omega', value=95.412, is_fixed=True)
            self.assertEqual(self.parameters['a_omega'].value, 95.412)
            self.assertTrue(self.parameters['a_omega'].is_fixed)

    def test_set_value(self):
        self.parameters.set_value('a_rho_prime', 0.654321)
        self.assertEqual(self.parameters['a_rho_prime'].value, 0.654321)

    def test_fix_parameters(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['mass_phi', 't_in_isoscalar'])
        with self.subTest():
            self.assertTrue(self.parameters['mass_phi'].is_fixed)
        with self.subTest():
            self.assertTrue(self.parameters['t_in_isoscalar'].is_fixed)

    def test_release_parameters(self):
        self.parameters.fix_parameters(['a_rho_double_prime'])
        self.parameters.release_parameters(['a_rho_double_prime'])
        self.assertFalse(self.parameters['a_rho_double_prime'].is_fixed)

    def test_release_all_parameters(self):
        self.parameters.fix_parameters(['a_phi', 'mass_omega_prime', 't_in_isovector'])
        self.parameters.release_all_parameters()
        self.assertFalse(self.parameters['a_phi'].is_fixed)
        self.assertFalse(self.parameters['mass_omega_prime'].is_fixed)
        self.assertFalse(self.parameters['t_in_isovector'].is_fixed)

    def test_fix_all_parameters(self):
        self.parameters.release_parameters(['a_phi', 'mass_omega_prime', 't_in_isovector'])
        self.parameters.fix_all_parameters()
        self.assertTrue(self.parameters['a_phi'].is_fixed)
        self.assertTrue(self.parameters['mass_omega_prime'].is_fixed)
        self.assertTrue(self.parameters['t_in_isovector'].is_fixed)

    def test_get_fixed_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['a_omega_double_prime', 'mass_rho_prime', 't_in_isovector'])
        self.assertListEqual(
            self.parameters.get_fixed_values(),
            [0.8, 0.15, 1.465],
        )

    def test_get_free_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_phi', 'mass_phi', 'decay_rate_phi',
            'a_phi_prime', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'a_omega', 'mass_omega', 'decay_rate_omega',
            'a_omega_prime', 'mass_omega_prime', 'decay_rate_omega_prime',
            'a_omega_double_prime', 'mass_omega_double_prime', 'decay_rate_omega_double_prime',
        ])
        self.assertListEqual(
            self.parameters.get_free_values(),
            [0.7, 0.8, 0.12, 0.77526, 0.1474, 0.13, 1.465, 0.4,
             0.14, 1.720, 0.25, 2.15, 0.3],
        )

    def test_get_all_values(self):
        self.assertListEqual(
            self.parameters.get_all_values(),
            [0.7, 0.8, 0.1, 0.78266, 0.00868, 0.2, 1.410, 0.29,
             0.15, 1.67, 0.315, 0.3, 1.019461, 0.004249, 0.35, 1.680, 0.150,
             2.159, 0.137, 0.12, 0.77526, 0.1474, 0.13, 1.465, 0.4,
             0.14, 1.720, 0.25, 2.15, 0.3],
        )

    def test_update_free_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_phi', 'mass_phi', 'decay_rate_phi',
            'a_phi_prime', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'a_omega', 'mass_omega', 'decay_rate_omega',
            'a_omega_prime', 'mass_omega_prime', 'decay_rate_omega_prime',
            'a_omega_double_prime', 'mass_omega_double_prime', 'decay_rate_omega_double_prime',
        ])
        self.parameters.update_free_values(
            [0.7, 0.82, 0.12, 0.7, 0.051, 0.0, 14.65, 0.4,
             0.24, 1.724, 0.25, 0.15, 7.4]
        )
        self.assertListEqual(
            self.parameters.get_free_values(),
            [0.7, 0.82, 0.12, 0.7, 0.051, 0.0, 14.65, 0.4,
             0.24, 1.724, 0.25, 0.15, 7.4]
        )

    def test_update_free_values__length_mismatch(self):
        self.parameters.release_all_parameters()
        with self.assertRaises(ValueError):
            self.parameters.update_free_values([1])

    def test___iter__(self):
        self.assertListEqual(
            [parameter.value for parameter in self.parameters],
            self.parameters.get_all_values(),
        )

    def test_get_bounds_for_free_parameters(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            't_in_isovector',
            'a_phi', 'decay_rate_phi',
            'a_phi_prime', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'a_omega', 'mass_omega', 'decay_rate_omega',
            'a_omega_prime', 'mass_omega_prime', 'decay_rate_omega_prime',
            'a_omega_double_prime', 'mass_omega_double_prime', 'decay_rate_omega_double_prime',
            'a_rho', 'mass_rho', 'decay_rate_rho',
            'a_rho_prime', 'mass_rho_prime', 'decay_rate_rho_prime',
            'mass_rho_triple_prime', 'decay_rate_rho_triple_prime',
        ])
        lower_bounds, upper_bounds = self.parameters.get_bounds_for_free_parameters()
        self.assertListEqual(
            lower_bounds,
            [0.5, 0.7071067811865476, -np.inf, 0.7745966692414834, 0.0],
        )
        self.assertListEqual(
            upper_bounds,
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        )
