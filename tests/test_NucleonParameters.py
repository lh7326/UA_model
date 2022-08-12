from unittest import TestCase
import numpy as np

from model_parameters.NucleonParameters import NucleonParameters, Parameter


class TestNucleonParameters(TestCase):

    def setUp(self):
        self.parameters = NucleonParameters(
            0.5, 0.6, 0.7, 0.8,
            1.5, 1.6, 1.7, 1.8,
            0.1, -0.1, 0.78266, 0.00868,
            0.2, 1.410, 0.29,
            1.67, 0.315,
            0.3, -0.3, 1.019461, 0.004249,
            0.35, -0.35, 1.680, 0.150,
            2.159, 0.137,
            0.12, 0.77526, 0.1474,
            1.465, 0.4,
            1.720, 0.25,
        )

    def test___getitem__(self):
        with self.subTest():
            par1 = self.parameters['a_dirac_rho']
            self.assertEqual(par1.value, 0.12)
        with self.subTest():
            par2 = self.parameters['mass_phi_prime']
            self.assertEqual(par2.value, 1.680)
        with self.subTest():
            par3 = self.parameters['t_in_pauli_isovector']
            self.assertEqual(par3.value, 1.8)
        with self.subTest():
            with self.assertRaises(KeyError):
                _ = self.parameters['non-existent-key']

    def test___setitem__(self):
        with self.subTest(msg='Testing type checking'):
            with self.assertRaises(TypeError):
                self.parameters['a_pauli_omega'] = 2.0  # type: ignore

        with self.subTest(msg='Testing parameter name checking'):
            with self.assertRaises(ValueError):
                self.parameters['a_pauli_omega'] = Parameter(name='new name', value=2.0, is_fixed=True)

        with self.subTest(msg='Testing a successful call'):
            self.parameters['a_pauli_omega'] = Parameter(name='a_pauli_omega', value=95.412, is_fixed=True)
            self.assertEqual(self.parameters['a_pauli_omega'].value, 95.412)
            self.assertTrue(self.parameters['a_pauli_omega'].is_fixed)

    def test_set_value(self):
        self.parameters.set_value('a_pauli_phi_prime', 0.654321)
        self.assertEqual(self.parameters['a_pauli_phi_prime'].value, 0.654321)

    def test_fix_parameters(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['mass_phi', 't_in_dirac_isoscalar'])
        with self.subTest():
            self.assertTrue(self.parameters['mass_phi'].is_fixed)
        with self.subTest():
            self.assertTrue(self.parameters['t_in_dirac_isoscalar'].is_fixed)

    def test_release_parameters(self):
        self.parameters.fix_parameters(['a_dirac_rho'])
        self.parameters.release_parameters(['a_dirac_rho'])
        self.assertFalse(self.parameters['a_dirac_rho'].is_fixed)

    def test_release_all_parameters(self):
        self.parameters.fix_parameters(['a_dirac_phi', 'mass_omega_prime', 't_in_pauli_isovector'])
        self.parameters.release_all_parameters()
        self.assertFalse(self.parameters['a_dirac_phi'].is_fixed)
        self.assertFalse(self.parameters['mass_omega_prime'].is_fixed)
        self.assertFalse(self.parameters['t_in_pauli_isovector'].is_fixed)

    def test_fix_all_parameters(self):
        self.parameters.release_parameters(['a_pauli_phi', 'mass_omega_prime', 't_in_dirac_isovector'])
        self.parameters.fix_all_parameters()
        self.assertTrue(self.parameters['a_pauli_phi'].is_fixed)
        self.assertTrue(self.parameters['mass_omega_prime'].is_fixed)
        self.assertTrue(self.parameters['t_in_dirac_isovector'].is_fixed)

    def test_get_fixed_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['a_dirac_omega_prime', 'mass_rho_prime', 't_in_dirac_isovector'])
        self.assertListEqual(
            self.parameters.get_fixed_values(),
            [0.5, 0.6, 0.8, 1.5, 1.6, 0.2, 1.465],
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
