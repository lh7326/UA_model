from unittest import TestCase
import cmath

from other_models import VMDModel


class TestVMDModel(TestCase):

    def test_add_vector_meson(self):
        vmd_model = VMDModel()
        vmd_model.add_vector_meson(1.0, 20.5)
        self.assertEqual(len(vmd_model._vector_mesons), 1)
        self.assertDictEqual(
            vmd_model._vector_mesons[0],
            {'coefficient': 1.0, 'mass': 20.5},
        )

    def test_create(self):
        vmd_model = VMDModel.create([{'coefficient': 1.3, 'mass': 12.4}, {'coefficient': 1.3, 'mass': 0.1}])
        self.assertEqual(len(vmd_model._vector_mesons), 2)
        self.assertEqual(vmd_model._vector_mesons[0]['coefficient'], 1.3)

    def test___call__(self):
        vmd_model = VMDModel.create([
            {'coefficient': 1.0, 'mass': 10.0},
            {'coefficient': 0.5, 'mass': 0.1},
        ])
        test_cases = [
            {'t': 0.0, 'expected_result': 1.5},
            {'t': 1.0, 'expected_result': 1.0050505050505052},
            {'t': 10j, 'expected_result': 0.9900995099004901 + 0.09950990049009551j},
        ]
        for case in test_cases:
            with self.subTest(case=case):
                self.assertTrue(cmath.isclose(
                    vmd_model(case['t']),
                    case['expected_result'],
                ))
