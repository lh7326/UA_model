from unittest import TestCase
import numpy as np

from model_parameters.KaonParametersPhiRatio import KaonParametersPhiRatio, Parameter


class TestModelParametersPhiRatio(TestCase):

    def setUp(self):
        self.parameters = KaonParametersPhiRatio(
            0.5, 0.6, 0.7, 0.8,
            0.1,  0.78266, 0.00868,
            0.2, 1.67,0.315,
            0.15, 0.28, 1.019461, 0.315,
            0.3, 1.67, 0.004249,
            2.159, 0.137,
            0.35, 0.775, 0.1474,
            0.159, 1.465,0.4,
            1.72, 0.25,
        )

    def test___getitem__(self):
        with self.subTest():
            par1 = self.parameters['a_rho_prime']
            self.assertEqual(par1.value, 0.159)
        with self.subTest():
            par2 = self.parameters['mass_phi_prime']
            self.assertEqual(par2.value, 1.67)
        with self.subTest():
            par3 = self.parameters['t_in_isovector']
            self.assertEqual(par3.value, 0.8)
        with self.subTest():
            with self.assertRaises(KeyError):
                _ = self.parameters['non-existent-key']

    def test___setitem__(self):
        with self.subTest(msg='Testing type checking'):
            with self.assertRaises(TypeError):
                self.parameters['a_rho'] = 2.0  # type: ignore

        with self.subTest(msg='Testing parameter name checking'):
            with self.assertRaises(ValueError):
                self.parameters['a_phi_charged'] = Parameter(name='new name', value=2.0, is_fixed=True)

        with self.subTest(msg='Testing a successful call'):
            self.parameters['a_phi_charged'] = Parameter(name='a_phi_charged', value=95.412, is_fixed=True)
            self.assertEqual(self.parameters['a_phi_charged'].value, 95.412)
            self.assertTrue(self.parameters['a_phi_charged'].is_fixed)

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
        self.parameters.fix_parameters(['a_rho_prime', 'a_phi_neutral'])
        self.parameters.release_parameters(['a_rho_prime', 'a_phi_neutral'])
        self.assertFalse(self.parameters['a_rho_prime'].is_fixed)
        self.assertFalse(self.parameters['a_phi_neutral'].is_fixed)

    def test_release_all_parameters(self):
        self.parameters.fix_parameters(['a_phi_charged', 'mass_rho_prime', 't_in_isovector'])
        self.parameters.release_all_parameters()
        self.assertFalse(self.parameters['a_phi_charged'].is_fixed)
        self.assertFalse(self.parameters['mass_rho_prime'].is_fixed)
        self.assertFalse(self.parameters['t_in_isovector'].is_fixed)

    def test_fix_all_parameters(self):
        self.parameters.release_parameters(['a_phi_neutral', 'mass_rho_prime', 't_in_isovector'])
        self.parameters.fix_all_parameters()
        self.assertTrue(self.parameters['a_phi_neutral'].is_fixed)
        self.assertTrue(self.parameters['mass_rho_prime'].is_fixed)
        self.assertTrue(self.parameters['t_in_isovector'].is_fixed)

    def test_get_fixed_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters(['mass_rho_prime', 't_in_isovector'])
        self.assertListEqual(
            self.parameters.get_fixed_values(),
            [0.5, 0.6, 0.8, 0.78266, 0.00868, 0.775, 0.1474, 1.465],
        )

    def test_get_free_values(self):
        self.parameters.release_all_parameters()
        self.parameters.fix_parameters([
            'a_phi_charged', 'mass_phi', 'decay_rate_phi',
            'a_phi_prime', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'mass_omega_double_prime', 'decay_rate_omega_double_prime',
        ])
        self.assertListEqual(
            self.parameters.get_free_values(),
            [0.7, 0.8, 0.1, 0.2, 0.28, 0.35, 0.159, 1.465, 0.4, 1.72, 0.25],
        )

    def test_get_all_values(self):
        self.assertListEqual(
            self.parameters.get_all_values(),
            [0.5, 0.6, 0.7, 0.8, 0.1,  0.78266, 0.00868,
             0.2, 1.67,0.315, 0.15, 0.28, 1.019461, 0.315,
             0.3, 1.67, 0.004249, 2.159, 0.137, 0.35, 0.775, 0.1474,
             0.159, 1.465,0.4, 1.72, 0.25],
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
            'a_phi_charged', 'a_phi_neutral', 'decay_rate_phi',
            'a_phi_prime', 'mass_phi_prime', 'decay_rate_phi_prime',
            'mass_phi_double_prime', 'decay_rate_phi_double_prime',
            'mass_omega', 'decay_rate_omega',
            'a_omega_double_prime', 'mass_omega_double_prime', 'decay_rate_omega_double_prime',
            'a_rho', 'mass_rho', 'decay_rate_rho',
            'a_rho_prime', 'mass_rho_prime', 'decay_rate_rho_prime',
            'mass_rho_double_prime', 'decay_rate_rho_double_prime',
        ])
        lower_bounds, upper_bounds = self.parameters.get_bounds_for_free_parameters(handpicked=False)
        self.assertListEqual(
            lower_bounds,
            [0.5, -np.inf, 0.7071067811865476],
        )
        self.assertListEqual(
            upper_bounds,
            [np.inf, np.inf, np.inf],
        )

    def test_from_list_to_list_consistency(self):
        list_of_parameters = [
            Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
            Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
            Parameter(name='t_in_isoscalar', value=1.443968455888137, is_fixed=False),
            Parameter(name='t_in_isovector', value=2.2187117617810133, is_fixed=True),
            Parameter(name='a_omega', value=2.5941212472427653, is_fixed=False),
            Parameter(name='mass_omega', value=1.262175646031432, is_fixed=True),
            Parameter(name='decay_rate_omega', value=0.29325211379153737, is_fixed=True),
            Parameter(name='a_omega_double_prime', value=-0.7584476425520864, is_fixed=False),
            Parameter(name='mass_omega_double_prime', value=1.65208370470636, is_fixed=True),
            Parameter(name='decay_rate_omega_double_prime', value=0.88444489573003, is_fixed=False),
            Parameter(name='a_phi_charged', value=-0.7613717224426919, is_fixed=False),
            Parameter(name='a_phi_neutral', value=0.37164182, is_fixed=True),
            Parameter(name='mass_phi', value=1.032465750329927, is_fixed=False),
            Parameter(name='decay_rate_phi', value=0.007187927884885847, is_fixed=False),
            Parameter(name='a_phi_prime', value=0.049186367861031886, is_fixed=True),
            Parameter(name='mass_phi_prime', value=1.798584035817393, is_fixed=False),
            Parameter(name='decay_rate_phi_prime', value=0.18684970951346638, is_fixed=False),
            Parameter(name='mass_phi_double_prime', value=2.1701948878602404, is_fixed=False),
            Parameter(name='decay_rate_phi_double_prime', value=0.018584914300650075, is_fixed=True),
            Parameter(name='a_rho', value=-0.004875792606689498, is_fixed=True),
            Parameter(name='mass_rho', value=3.463509700968759, is_fixed=True),
            Parameter(name='decay_rate_rho', value=1.4560004801176234, is_fixed=True),
            Parameter(name='a_rho_prime', value=-0.26856897443026406, is_fixed=True),
            Parameter(name='mass_rho_prime', value=1.6384823897203615, is_fixed=False),
            Parameter(name='decay_rate_rho_prime', value=0.13857062014248653, is_fixed=False),
            Parameter(name='mass_rho_double_prime', value=2.245836486713978, is_fixed=False),
            Parameter(name='decay_rate_rho_double_prime', value=0.1035328301790177, is_fixed=True),
        ]
        model_parameters = KaonParametersPhiRatio.from_list(list_of_parameters)
        self.assertIsInstance(model_parameters, KaonParametersPhiRatio)
        self.assertListEqual(list_of_parameters, model_parameters.to_list())

    def test_to_list_from_list_consistency(self):
        recreated = KaonParametersPhiRatio.from_list(self.parameters.to_list())
        self.assertEqual(
            recreated['a_rho_prime'].value,
            self.parameters['a_rho_prime'].value,
        )
        self.assertEqual(
            recreated['mass_phi_prime'].value,
            self.parameters['mass_phi_prime'].value,
        )
        self.assertEqual(
            recreated['t_in_isovector'].value,
            self.parameters['t_in_isovector'].value,
        )
