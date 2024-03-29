from configparser import ConfigParser

from kaon_production.data import read_data, KaonDatapoint
from model_parameters import KaonParameters
from task.kaon_cross_section_tasks import TaskFixAccordingToParametersFit


def make_parameters(t_0_isoscalar, t_0_isovector, list_of_values):
    return KaonParameters.from_ordered_values(list_of_values, t_0_isoscalar, t_0_isovector)


def prepare_data(ts_charged, css_charged, errors_charged, ts_neutral, css_neutral, errors_neutral):
    ts = [KaonDatapoint(t, True) for t in ts_charged]
    cross_sections = list(css_charged)
    errors = list(errors_charged)
    ts += [KaonDatapoint(t, False) for t in ts_neutral]
    cross_sections += list(css_neutral)
    errors += list(errors_neutral)

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

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    path_to_reports = '/home/lukas/reports'

    charged_ts, charged_cross_sections_values, charged_errors = read_data('charged_new_data2.csv')
    neutral_ts, neutral_cross_sections_values, neutral_errors = read_data('neutral_kaon.csv')

    parameters = make_parameters(
        t_0_isoscalar, t_0_isovector,
        list_of_values=[4.104732671275326, 6.945876504806479, 0.77526, 1.465, 1.72, 2.15, 0.1474, 0.4, 0.5220413015662677, 0.9262000511188161, 0.5579553419980545, -0.06608398477649186, 0.19847722499260162, 0.78266, 1.0190415494118024, 1.41, 1.67, 1.68, 2.159, 0.00868, 0.004249, 0.9228912480513007, 0.315, 0.15, 0.137, 0.20006026821487782, 0.3317378089249629, 0.14633274655460118, -0.17367426366140964, -0.010680398179201966],
    )

    ts, cross_sections, errors = prepare_data(
        charged_ts, charged_cross_sections_values, charged_errors,
        neutral_ts, neutral_cross_sections_values, neutral_errors,
    )

    parameters.set_value('decay_rate_phi_prime', 0.150)
    parameters.release_all_parameters()
    parameters.fix_resonances()
    parameters.release_parameters([
        't_in_isoscalar', 't_in_isovector',
        'decay_rate_omega_prime',
        'mass_phi', 'decay_rate_phi', 'mass_phi_prime', 'decay_rate_phi_prime', 'mass_phi_double_prime', 'decay_rate_phi_double_prime',
        'decay_rate_rho_double_prime', 'decay_rate_rho_triple_prime',
    ])
    task = TaskFixAccordingToParametersFit(
        'Fit', parameters,
        ts, cross_sections, errors,
        kaon_mass, alpha, hc_squared,
        None, True, use_handpicked_bounds=False
    )
    print(task.run().get_ordered_values())
    print(task.report)
