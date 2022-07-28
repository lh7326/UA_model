from configparser import ConfigParser

from kaon_production.data import read_data, Datapoint
from kaon_production.utils import make_partial_cross_section_for_parameters
from model_parameters import KaonParameters, KaonParametersSimplified, Parameter
from plotting.plot_fit import plot_cs_fit_neutral_plus_charged


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

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    charged_ts, charged_cross_sections_values, charged_errors = read_data('charged_kaon.csv')
    neutral_ts, neutral_cross_sections_values, neutral_errors = read_data('neutral_kaon.csv')
    ts, css, errs = prepare_data(charged_ts, charged_cross_sections_values,charged_errors,
                                 neutral_ts, neutral_cross_sections_values, neutral_errors)

    parameters = KaonParameters.from_list([
        Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
        Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
        Parameter(name='t_in_isoscalar', value=1.0555688365879876, is_fixed=False),
        Parameter(name='t_in_isovector', value=2.8196561781509173, is_fixed=True),
        Parameter(name='a_omega', value=0.0, is_fixed=True),
        Parameter(name='mass_omega', value=0.7899999999999999, is_fixed=True),
        Parameter(name='decay_rate_omega', value=0.0088999999995, is_fixed=True),
        Parameter(name='a_omega', value=0.2856684158418351, is_fixed=True),
        Parameter(name='mass_omega', value=0.7899999999999999, is_fixed=True),
        Parameter(name='decay_rate_omega', value=0.0088999999995, is_fixed=True),
        Parameter(name='a_omega_prime', value=-0.10101371171936338, is_fixed=True),
        Parameter(name='mass_omega_prime', value=1.4499999999990802, is_fixed=True),
        Parameter(name='decay_rate_omega_prime', value=0.32999999999938134, is_fixed=True),
        Parameter(name='a_omega_double_prime', value=0.1694783292593881, is_fixed=True),
        Parameter(name='mass_omega_double_prime', value=1.6749999999991625, is_fixed=True),
        Parameter(name='decay_rate_omega_double_prime', value=0.3299999999995, is_fixed=True),
        Parameter(name='a_phi', value=0.27371704966529914, is_fixed=True),
        Parameter(name='mass_phi', value=1.019283756363553, is_fixed=True),
        Parameter(name='decay_rate_phi', value=0.004160766514697237, is_fixed=True),
        Parameter(name='a_phi_prime', value=-0.14151659390407614, is_fixed=True),
        Parameter(name='mass_phi_prime', value=1.6750000000008376, is_fixed=False),
        Parameter(name='decay_rate_phi_prime', value=0.19999999999847543, is_fixed=True),
        Parameter(name='mass_phi_double_prime', value=2.1999999999996596, is_fixed=True),
        Parameter(name='decay_rate_phi_double_prime', value=0.19999999999949925, is_fixed=True),
        #Parameter(name='a_rho', value=0.0, is_fixed=True),
        #Parameter(name='mass_rho', value=0.7799999999999999, is_fixed=True),
        #Parameter(name='decay_rate_rho', value=0.14999999999998914, is_fixed=True),
        Parameter(name='a_rho', value=0.5657394685768862, is_fixed=True),
        Parameter(name='mass_rho', value=0.7799999999999999, is_fixed=True),
        Parameter(name='decay_rate_rho', value=0.14999999999998914, is_fixed=True),
        Parameter(name='a_rho_prime', value=-0.022826343180193855, is_fixed=True),
        Parameter(name='mass_rho_prime', value=1.479999999999085, is_fixed=False),
        Parameter(name='decay_rate_rho_prime', value=0.42959950613649417, is_fixed=False),
        Parameter(name='a_rho_double_prime', value=-0.07766237555900325, is_fixed=True),
        Parameter(name='mass_rho_double_prime', value=1.6800000000008701, is_fixed=False),
        Parameter(name='decay_rate_rho_double_prime', value=0.3109833428610703, is_fixed=True),
        Parameter(name='mass_rho_triple_prime', value=2.0000000000000213, is_fixed=True),
        Parameter(name='decay_rate_rho_triple_prime', value=0.18040126977857288, is_fixed=True)
    ])
    parameters.fix_all_parameters()
    f = make_partial_cross_section_for_parameters(kaon_mass, alpha, hc_squared, parameters)
    plot_cs_fit_neutral_plus_charged(ts, css, errs, f, (), 'plot_fit', show=True, save_dir=None)
