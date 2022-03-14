from unittest import TestCase
import cmath
from configparser import ConfigParser

from ua_model.KaonUAModel import KaonUAModel


class TestKaonUAModel(TestCase):

    def test___call__(self):

        config = ConfigParser()
        config['branch_points'] = {
            'kaon_isoscalar': '1.0',
            'kaon_isovector': '0.1',
        }
        config['resonances'] = {
            'omega_mass': '1.4',
            'omega_decay_rate': '0.001',
            'omega_prime_mass': '1.5',
            'omega_prime_decay_rate': '0.001',
            'omega_double_prime_mass': '1.6',
            'omega_double_prime_decay_rate': '0.001',
            'phi_mass': '2.0',
            'phi_decay_rate': '0.01',
            'phi_prime_mass': '2.2',
            'phi_prime_decay_rate': '0.01',
            'phi_double_prime_mass': '2.4',
            'phi_double_prime_decay_rate': '0.01',
            'rho_mass': '3.1',
            'rho_decay_rate': '0.1',
            'rho_prime_mass': '3.2',
            'rho_prime_decay_rate': '0.1',
            'rho_double_prime_mass': '3.3',
            'rho_double_prime_decay_rate': '0.1',
        }

        kaon_model = KaonUAModel(
            a_omega=0.21,
            a_omega_prime=0.09,
            a_omega_double_prime=0.12,
            a_phi=0.15,
            a_phi_prime=0.07,
            a_rho=0.34,
            a_rho_prime=0.03,
            t_in_isoscalar=4.5,
            t_in_isovector=14.7,
            config=config,
            charged_variant=True,
        )

        with self.subTest(msg='Testing "charged_variant" flag'):
            self.assertTrue(kaon_model.charged_variant)

        with self.subTest(msg='Test normalization (t=0) --- the variant K+ K-'):
            expected = 1.0
            kaon_model.charged_variant = True
            actual = kaon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected))

        with self.subTest(msg='Test normalization (t=0) --- the variant K0 anti-K0'):
            expected = 0.0
            kaon_model.charged_variant = False
            actual = kaon_model(0.0)
            self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))

        test_cases = [
            {'t': 1.7, 'charged_variant': True, 'expected_value': 3.07377941981359-0.02693634055780967j},
            {'t': 1.7, 'charged_variant': False, 'expected_value': 1.8749940882607046+0.0037825009468283807j},
            {'t': 0.4+1.2j, 'charged_variant': True, 'expected_value': 0.9476115946337496+0.3317430734118266j},
            {'t': 162.42-0.647j, 'charged_variant': False,
             'expected_value': 0.027572466270023334+0.00011626299922461292j},
            {'t': 84.1-9124.1j, 'charged_variant': True,
             'expected_value': -5.668628530153822e-06-0.0006860125397040155j},
            {'t': 62.4j, 'charged_variant': False, 'expected_value': -0.01280598718730843-0.06423566902122738j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                kaon_model.charged_variant = case['charged_variant']
                actual = kaon_model(case['t'])
                expected = case['expected_value']
                print(actual, expected)
                self.assertTrue(cmath.isclose(actual, expected, abs_tol=1.0e-15))
