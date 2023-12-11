from configparser import ConfigParser

from nucleon_production.data import read_data, NucleonDatapoint
from model_parameters import NucleonParameters
from task.nucleon_cross_section_tasks import TaskFixAccordingToParametersFit


def make_parameters(
        nucleon_mass, magnetic_moment_proton, magnetic_moment_neutron,
        t_0_dirac_isoscalar, t_0_dirac_isovector,
        t_0_pauli_isoscalar, t_0_pauli_isovector, list_of_values):
    return NucleonParameters.from_ordered_values(
        list_of_values, nucleon_mass=nucleon_mass, magnetic_moment_proton=magnetic_moment_proton,
        magnetic_moment_neutron=magnetic_moment_neutron, t_0_dirac_isoscalar=t_0_dirac_isoscalar,
        t_0_dirac_isovector=t_0_dirac_isovector, t_0_pauli_isoscalar=t_0_pauli_isoscalar,
        t_0_pauli_isovector=t_0_pauli_isovector)


def prepare_data(ts_proton, css_proton, errors_proton, ts_neutron, css_neutron, errors_neutron):
    ts = [NucleonDatapoint(t, True, True) for t in ts_proton]
    cross_sections = list(css_proton)
    errors = list(errors_proton)
    ts += [NucleonDatapoint(t, False, True) for t in ts_neutron]
    cross_sections += list(css_neutron)
    errors += list(errors_neutron)

    return zip(
        *sorted(
            zip(ts, cross_sections, errors),
            key=lambda tup: tup[0].t,
        )
    )


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    proton_mass = config.getfloat('constants', 'proton_mass')
    magnetic_moment_proton = config.getfloat('constants', 'proton_magnetic_moment')
    magnetic_moment_neutron = config.getfloat('constants', 'neutron_magnetic_moment')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports'

    ts, css, errors = read_data('proton_cs_extended.csv')

    list_of_values = [1.921505175780696, 1.5790639230453043, 1.9520233934653979, 1.4222095291234824, 0.5392755402487763, 0.008486878507497419, 0.19466200713482615, -0.00875921846784497, 0.15592815589971398, 0.19825369592717146, -0.006864173123697644, -0.09856559678755873, 0.78266, 1.019461, 1.41, 1.67, 1.68, 2.159, 0.00868, 0.004249, 0.29, 0.315, 0.15, 0.137, 0.77526, 1.465, 1.72, 0.1474, 0.4, 0.25]

    parameters = make_parameters(
        proton_mass, magnetic_moment_proton, magnetic_moment_neutron,
        t_0_isoscalar, t_0_isovector, t_0_isoscalar, t_0_isovector,
        list_of_values=list_of_values,
    )

    ts, css, errors = prepare_data(
        ts, css, errors,
        [], [], [],
    )
    parameters.fix_resonances()
    task = TaskFixAccordingToParametersFit(
        'Full Fit', parameters,
        ts, css, errors,
        proton_mass, alpha, hc_squared,
        None, True, use_handpicked_bounds=False
    )
    print(task.run().get_ordered_values())
    print(task.report)
