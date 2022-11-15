from configparser import ConfigParser

from multiprocessing import Pool

from nucleon_production.data import read_data
from model_parameters.NucleonParameters import NucleonParameters
from pipeline.NucleonCrossSectionIterativePipeline import NucleonCrossSectionIterativePipeline
from common.utils import perturb_model_parameters


def make_initial_parameters(
        nucleon_mass, proton_magnetic_moment, neutron_magnetic_moment,
        t_0_dirac_isoscalar, t_0_dirac_isovector,
        t_0_pauli_isoscalar, t_0_pauli_isovector
    ):
    return NucleonParameters(
        nucleon_mass=nucleon_mass,
        magnetic_moment_proton=proton_magnetic_moment,
        magnetic_moment_neutron=neutron_magnetic_moment,
        t_0_dirac_isoscalar=t_0_dirac_isoscalar,
        t_0_dirac_isovector=t_0_dirac_isovector,
        t_0_pauli_isoscalar=t_0_pauli_isoscalar,
        t_0_pauli_isovector=t_0_pauli_isovector,
        t_in_dirac_isoscalar=1.0,
        t_in_dirac_isovector=2.0,
        t_in_pauli_isoscalar=1.0,
        t_in_pauli_isovector=2.0,
        a_dirac_omega=1.0,
        a_pauli_omega=-0.2,
        mass_omega=0.78266,
        decay_rate_omega=0.00868,
        a_dirac_omega_prime=0.2,
        mass_omega_prime=1.410,
        decay_rate_omega_prime=0.29,
        mass_omega_double_prime=1.670,
        decay_rate_omega_double_prime=0.315,
        a_dirac_phi=-1.0,
        a_pauli_phi=0.2,
        mass_phi=1.019461,
        decay_rate_phi=0.004249,
        a_dirac_phi_prime=0.2,
        a_pauli_phi_prime=0.2,
        mass_phi_prime=1.680,
        decay_rate_phi_prime=0.150,
        mass_phi_double_prime=2.159,
        decay_rate_phi_double_prime=0.137,
        a_dirac_rho=0.4,
        mass_rho=0.76388,
        decay_rate_rho=0.14428,
        mass_rho_prime=1.34231,
        decay_rate_rho_prime=0.49217,
        mass_rho_double_prime=1.7185,
        decay_rate_rho_double_prime=0.48958,
    )


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    proton_mass = config.getfloat('constants', 'proton_mass')
    proton_magnetic_moment = config.getfloat('constants', 'proton_magnetic_moment')
    neutron_magnetic_moment = config.getfloat('constants', 'neutron_magnetic_moment')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports'

    ts_proton_electric, css_proton_electric, errors_proton_electric = read_data('charged_kaon.csv')


    def f(name):
        initial_parameters = make_initial_parameters(
            nucleon_mass=proton_mass,
            proton_magnetic_moment=proton_magnetic_moment,
            neutron_magnetic_moment=neutron_magnetic_moment,
            t_0_dirac_isoscalar=t_0_isoscalar,
            t_0_dirac_isovector=t_0_isovector,
            t_0_pauli_isoscalar=t_0_isoscalar,
            t_0_pauli_isovector=t_0_isovector,
        )

        initial_parameters = perturb_model_parameters(
            initial_parameters,
            perturbation_size=0.5, perturbation_size_resonances=0.2,
            use_handpicked_bounds=True,
        )
        numbers = (5, 4, 6, 2, 8, 4, 10, 17, 5)
        repetitions = (5, 20, 20, 10, 40, 10, 20, 20, 10)
        pipeline = NucleonCrossSectionIterativePipeline(
            name, initial_parameters,
            ts_proton_electric[160:], css_proton_electric[160:], errors_proton_electric[160:],
            [], [], [],
            [], [], [],
            [], [], [],
            proton_mass, alpha, hc_squared,
            path_to_reports, plot=False, use_handpicked_bounds=True,
            nr_free_params=numbers, nr_iterations=repetitions,
            nr_initial_rounds_with_fixed_resonances=5,
        )
        return pipeline.run()

    final_results = []
    best_fit = {'chi_squared': None, 'name': None, 'parameters': None}
    for i in range(2):
        result = f(f'test_{i}')
        print(result)

        if result and result.get('chi_squared', None) is not None:
            final_results.append(result)
            if best_fit['chi_squared'] is None or result['chi_squared'] < best_fit['chi_squared']:
                best_fit = result
    print('Best fit: ', best_fit)

    for final_result in sorted(final_results, key=lambda fr: fr['chi_squared'])[:3]:
        print(final_result)
