from configparser import ConfigParser

from kaon_production.data import read_data, KaonDatapoint
from common.utils import make_partial_form_factor_for_parameters, make_partial_cross_section_for_parameters
from model_parameters import KaonParametersB, Parameter
from plotting.plot_fit import plot_ff_fit_neutral_plus_charged
from task.ResidualOscillationsTask import ResidualOscillationsTask


def prepare_data(ts_charged, css_charged, errors_charged, ts_neutral, css_neutral, errors_neutral):
    ts = [KaonDatapoint(t, True, True) for t in ts_charged]
    cross_sections = list(css_charged)
    errors = list(errors_charged)
    ts += [KaonDatapoint(t, False, True) for t in ts_neutral]
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

    charged_ts, charged_cross_sections_values, charged_errors = read_data('charged_ff_2.csv', '')
    ts, ffs, errs = prepare_data(charged_ts, charged_cross_sections_values, charged_errors, [], [], [])

    # parameters = KaonParametersB.from_ordered_values([
    #     0.9392300073668993, 1.0843211231100962, 0.7597833518800249, 1.5382094504664614, 1.7735588154198763,
    #     0.16077299740026443, 0.40224673665593264, 0.252229778815626, 0.389265219800551, 0.017898033984832362,
    #     0.7390610355192804, -2.92865863357512, 1.6958225009234162, 1.0190481914084335, 2.4746696999190707,
    #     0.014220156941473517, 0.44554987184578027, 0.3406412462563918, -0.004188802154015758, 4.16507760424825,
    #     0.4794865915790986, -0.005663957075199569, -0.20861912682917905, 0.3336407756472722
    # ], t_0_isoscalar=0.17531904388276887, t_0_isovector=0.07791957505900839)
    parameters = KaonParametersB.from_list([
        Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
        Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
        Parameter(name='t_in_isoscalar', value=3.2291595069331067, is_fixed=False),
        Parameter(name='t_in_isovector', value=0.5835126419012053, is_fixed=False),
        Parameter(name='a_omega', value=0.32729801061391034, is_fixed=False),
        Parameter(name='mass_omega', value=0.78266, is_fixed=True),
        Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
        Parameter(name='a_omega_double_prime', value=-0.1322560747876571, is_fixed=False),
        Parameter(name='mass_omega_double_prime', value=1.67, is_fixed=True),
        Parameter(name='decay_rate_omega_double_prime', value=0.315, is_fixed=True),
        Parameter(name='a_phi', value=0.33178701236457975, is_fixed=False),
        Parameter(name='mass_phi', value=1.0190536833008472, is_fixed=False),
        Parameter(name='decay_rate_phi', value=0.004229702507790191, is_fixed=False),
        Parameter(name='a_phi_prime', value=0.08303027600714324, is_fixed=False),
        Parameter(name='mass_phi_prime', value=3.0321618703565654, is_fixed=False),
        Parameter(name='decay_rate_phi_prime', value=1.7409626508527738, is_fixed=False),
        Parameter(name='mass_phi_double_prime', value=1.984329537423646, is_fixed=False),
        Parameter(name='decay_rate_phi_double_prime', value=0.6816716770166859, is_fixed=False),
        Parameter(name='a_rho', value=0.5456549089807899, is_fixed=False),
        Parameter(name='mass_rho', value=0.76388, is_fixed=True),
        Parameter(name='decay_rate_rho', value=0.14428, is_fixed=True),
        Parameter(name='a_rho_prime', value=-0.03127769956987407, is_fixed=False),
        Parameter(name='mass_rho_prime', value=1.32635, is_fixed=True),
        Parameter(name='decay_rate_rho_prime', value=0.32413, is_fixed=True),
        Parameter(name='mass_rho_double_prime', value=1.77054, is_fixed=True),
        Parameter(name='decay_rate_rho_double_prime', value=0.26898, is_fixed=True)]
    )
    parameters.fix_all_parameters()
    f = make_partial_form_factor_for_parameters(parameters)
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    g = make_partial_cross_section_for_parameters(
        alpha=alpha, hc_squared=hc_squared, parameters=parameters,
        charged_kaon_mass=charged_kaon_mass, neutral_kaon_mass=neutral_kaon_mass)
    print(f([KaonDatapoint(t=8.0, is_charged=True, is_for_cross_section=True)]))
    print(g([KaonDatapoint(t=8.0, is_charged=True, is_for_cross_section=True)]))
    plot_ff_fit_neutral_plus_charged(ts, ffs, errs, f, (), 'plot_fit', show=True, save_dir=None)

    task = ResidualOscillationsTask(
        'ResidualOscillationsTask', parameters, ts, ffs, errs,
        product_particle_mass=charged_kaon_mass, alpha=alpha, hc_squared=hc_squared,
        reports_dir='/home/lukas/reports/ff', plot=True
    )
    task.run()
