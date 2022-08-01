from configparser import ConfigParser
from typing import Callable, List, Union, Tuple

from kaon_production.data import read_data, Datapoint
from cross_section.ScalarMesonProductionTotalCrossSection import ScalarMesonProductionTotalCrossSection
from ua_model.SingleComponentModel import SingleComponentModel
from plotting.plot_fit import plot_cs_fit_neutral_plus_charged


def make_function_cross_section(
        k_meson_mass: float,
        alpha: float,
        hc_squared: float,
        a: float,
        mass: float,
        decay_rate: float,
        t_0: float,
        t_in: float,
        ) -> Callable[[List[Union[Datapoint, Tuple[float, float]]]], List[complex]]:

    ff_model = SingleComponentModel(t_0, t_in, a, mass, decay_rate)
    config = ConfigParser()
    config['constants'] = {'alpha': alpha, 'hc_squared': hc_squared}
    cross_section_model = ScalarMesonProductionTotalCrossSection(k_meson_mass, ff_model, config)

    def f(ts: List[Union[Datapoint, Tuple[float, float]]]) -> List[complex]:
        results = []
        for datapoint in ts:
            if isinstance(datapoint, Datapoint):
                cross_section_model.form_factor.charged_variant = datapoint.is_charged
                results.append(cross_section_model(datapoint.t))
            else:
                cross_section_model.form_factor.charged_variant = bool(datapoint[1])
                results.append(cross_section_model(datapoint[0]))
        return results
    return f


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

    f = make_function_cross_section(
        kaon_mass, alpha, hc_squared,
        a=1.0, mass=0.78266, decay_rate=0.00868,
        t_0=0.17531904388276887, t_in=1.0555688365879876,
    )

    plot_cs_fit_neutral_plus_charged(ts, css, errs, f, (), 'phi', show=True, save_dir=None)
