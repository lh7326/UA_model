from configparser import ConfigParser

from kaon_production.data import read_data, Datapoint
from kaon_production.utils import make_partial_form_factor_for_parameters, make_partial_cross_section_for_parameters
from model_parameters import KaonParametersB, Parameter
from plotting.plot_fit import plot_ff_fit_neutral_plus_charged


def prepare_data(ts_charged, css_charged, errors_charged, ts_neutral, css_neutral, errors_neutral):
    ts = [Datapoint(t, True) for t in ts_charged]
    cross_sections = list(css_charged)
    errors = list(errors_charged)
    ts += [Datapoint(t, False) for t in ts_neutral]
    cross_sections += list(css_neutral)
    errors += list(errors_neutral)

    ts, cross_sections, errors = zip(
        *sorted(
            zip(ts, cross_sections, errors),
            key=lambda tup: tup[0].t,
        )
    )
    return ts, cross_sections, errors


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    charged_ts, charged_cross_sections_values, charged_errors = read_data('charged_ff_2.csv')
    ts, ffs, errs = prepare_data(charged_ts, charged_cross_sections_values, charged_errors, [], [], [])

    parameters = KaonParametersB.from_list(
        [Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
         Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
         Parameter(name='t_in_isoscalar', value=3.235342095408974, is_fixed=True),
         Parameter(name='t_in_isovector', value=0.5835126420559626, is_fixed=False),
         Parameter(name='a_omega', value=0.3246694673250383, is_fixed=True),
         Parameter(name='mass_omega', value=0.78266, is_fixed=True),
         Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
         Parameter(name='a_omega_double_prime', value=-0.1321384421882062, is_fixed=False),
         Parameter(name='mass_omega_double_prime', value=1.67, is_fixed=True),
         Parameter(name='decay_rate_omega_double_prime', value=0.315, is_fixed=True),
         Parameter(name='a_phi', value=0.3318718476115985, is_fixed=False),
         Parameter(name='mass_phi', value=1.0190545080055673, is_fixed=True),
         Parameter(name='decay_rate_phi', value=0.004231176612977179, is_fixed=False),
         Parameter(name='a_phi_prime', value=0.0912398416941766, is_fixed=False),
         Parameter(name='mass_phi_prime', value=2.9707977398726992, is_fixed=True),
         Parameter(name='decay_rate_phi_prime', value=1.7562905421961654, is_fixed=True),
         Parameter(name='mass_phi_double_prime', value=1.9742425823655207, is_fixed=False),
         Parameter(name='decay_rate_phi_double_prime', value=0.6973339737579703, is_fixed=True),
         Parameter(name='a_rho', value=0.5457284708514346, is_fixed=False),
         Parameter(name='mass_rho', value=0.76388, is_fixed=True),
         Parameter(name='decay_rate_rho', value=0.14428, is_fixed=True),
         Parameter(name='a_rho_prime', value=-0.0323036090003582, is_fixed=False),
         Parameter(name='mass_rho_prime', value=1.32635, is_fixed=True),
         Parameter(name='decay_rate_rho_prime', value=0.32413, is_fixed=True),
         Parameter(name='mass_rho_double_prime', value=1.77054, is_fixed=True),
         Parameter(name='decay_rate_rho_double_prime', value=0.26898, is_fixed=True)]
    )
    parameters.fix_all_parameters()
    f = make_partial_form_factor_for_parameters(parameters)
    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    g = make_partial_cross_section_for_parameters(
        k_meson_mass=kaon_mass, alpha=alpha, hc_squared=hc_squared, parameters=parameters)
    print(f([Datapoint(t=1.7, is_charged=True)]))
    print(g([Datapoint(t=1.7, is_charged=True)]))
    plot_ff_fit_neutral_plus_charged(ts, ffs, errs, f, (), 'plot_fit', show=True, save_dir=None)
