from configparser import ConfigParser
from typing import List, Union

from kaon_production.data import read_data, Datapoint
from model_parameters import Parameter, KaonParameters, KaonParametersSimplified, KaonParametersFixedRhoOmega
from kaon_production.utils import make_partial_cross_section_for_parameters


def calculate_deviations_from_fit(
        parameters: Union[KaonParameters, KaonParametersSimplified, KaonParametersFixedRhoOmega],
        ts: List[float],
        cross_section_values: List[float],
        errors: List[float],
        charged_kaons: bool,
        kaon_mass: float,
        alpha: float,
        hc_squared: float,
):
    data = [Datapoint(t=t, is_charged=charged_kaons) for t in ts]
    parameters.fix_all_parameters()
    f = make_partial_cross_section_for_parameters(kaon_mass, alpha, hc_squared, parameters)
    fit = f(data)
    return [
        (t, error, cs, fit_val, abs(cs - fit_val) / error)
        for t, error, cs, fit_val in zip(ts, errors, cross_section_values, fit)
    ]


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')

    charged_ts, charged_cross_sections_values, charged_errors = read_data(
        'charged_kaon_cropped_manually.csv')
    neutral_ts, neutral_cross_sections_values, neutral_errors = read_data(
        'neutral_kaon_cropped_manually.csv')

    # parameters = KaonParametersFixedRhoOmega.from_list([
    #     Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True),
    #     Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True),
    #     Parameter(name='t_in_isoscalar', value=1.8772034161385576, is_fixed=False),
    #     Parameter(name='t_in_isovector', value=0.0779195751589424, is_fixed=False),
    #     Parameter(name='a_omega', value=0.24153839198896812, is_fixed=False),
    #     Parameter(name='mass_omega', value=0.78266, is_fixed=True),
    #     Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True),
    #     Parameter(name='a_omega_prime', value=0.019668419184461317, is_fixed=False),
    #     Parameter(name='mass_omega_prime', value=1.3701107120038778, is_fixed=False),
    #     Parameter(name='decay_rate_omega_prime', value=0.3299999579784306, is_fixed=False),
    #     Parameter(name='a_omega_double_prime', value=-0.06010453788953642, is_fixed=False),
    #     Parameter(name='mass_omega_double_prime', value=1.6650000001658132, is_fixed=False),
    #     Parameter(name='decay_rate_omega_double_prime', value=0.3000034503207487, is_fixed=False),
    #     Parameter(name='a_phi', value=0.2968326902510261, is_fixed=False),
    #     Parameter(name='mass_phi', value=1.0202211330285356, is_fixed=False),
    #     Parameter(name='decay_rate_phi', value=0.004497708709693714, is_fixed=False),
    #     Parameter(name='a_phi_prime', value=-0.012705117132072041, is_fixed=False),
    #     Parameter(name='mass_phi_prime', value=1.6750000001662304, is_fixed=False),
    #     Parameter(name='decay_rate_phi_prime', value=0.10841423532575731, is_fixed=False),
    #     Parameter(name='mass_phi_double_prime', value=2.194376168264091, is_fixed=False),
    #     Parameter(name='decay_rate_phi_double_prime', value=0.19999915005790322, is_fixed=False),
    #     Parameter(name='a_rho', value=0.5639957943258546, is_fixed=False),
    #     Parameter(name='mass_rho', value=0.77526, is_fixed=True),
    #     Parameter(name='decay_rate_rho', value=0.1474, is_fixed=True),
    #     Parameter(name='a_rho_prime', value=-0.08798792709813609, is_fixed=False),
    #     Parameter(name='mass_rho_prime', value=1.4799999998518092, is_fixed=False),
    #     Parameter(name='decay_rate_rho_prime', value=0.3471541745393237, is_fixed=False),
    #     Parameter(name='a_rho_double_prime', value=-0.009628247264942328, is_fixed=False),
    #     Parameter(name='mass_rho_double_prime', value=1.6800000001673372, is_fixed=False),
    #     Parameter(name='decay_rate_rho_double_prime', value=0.15976993441741424, is_fixed=False),
    #     Parameter(name='mass_rho_triple_prime', value=2.0019217068106614, is_fixed=False),
    #     Parameter(name='decay_rate_rho_triple_prime', value=0.17754424543070346, is_fixed=False)
    # ])
    parameters = KaonParametersFixedRhoOmega.from_list([Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True), Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True), Parameter(name='t_in_isoscalar', value=1.0365569328792128, is_fixed=True), Parameter(name='t_in_isovector', value=1.6795999919783429, is_fixed=True), Parameter(name='a_omega', value=0.3640295777639465, is_fixed=True), Parameter(name='mass_omega', value=0.78266, is_fixed=True), Parameter(name='decay_rate_omega', value=0.00868, is_fixed=True), Parameter(name='a_omega_prime', value=-0.04420232725126961, is_fixed=True), Parameter(name='mass_omega_prime', value=1.4499999999999997, is_fixed=True), Parameter(name='decay_rate_omega_prime', value=0.3299999999995, is_fixed=True), Parameter(name='a_omega_double_prime', value=-0.004446306776718244, is_fixed=True), Parameter(name='mass_omega_double_prime', value=1.6650000000008336, is_fixed=True), Parameter(name='decay_rate_omega_double_prime', value=0.3000000001, is_fixed=True), Parameter(name='a_phi', value=0.21515490214604754, is_fixed=False), Parameter(name='mass_phi', value=1.0174220440395478, is_fixed=False), Parameter(name='decay_rate_phi', value=0.004499999999997257, is_fixed=True), Parameter(name='a_phi_prime', value=-0.04457599206853455, is_fixed=True), Parameter(name='mass_phi_prime', value=1.6750000000000003, is_fixed=False), Parameter(name='decay_rate_phi_prime', value=0.1512945056878342, is_fixed=True), Parameter(name='mass_phi_double_prime', value=2.1941336183231037, is_fixed=True), Parameter(name='decay_rate_phi_double_prime', value=0.19999999999949045, is_fixed=True), Parameter(name='a_rho', value=0.5858562425364882, is_fixed=False), Parameter(name='mass_rho', value=0.77526, is_fixed=True), Parameter(name='decay_rate_rho', value=0.1474, is_fixed=True), Parameter(name='a_rho_prime', value=-0.14577364351226266, is_fixed=True), Parameter(name='mass_rho_prime', value=1.4799999999999998, is_fixed=True), Parameter(name='decay_rate_rho_prime', value=0.39522683844974404, is_fixed=True), Parameter(name='a_rho_double_prime', value=0.02049285609676272, is_fixed=True), Parameter(name='mass_rho_double_prime', value=1.7399999999999973, is_fixed=False), Parameter(name='decay_rate_rho_double_prime', value=0.5999999999999913, is_fixed=True), Parameter(name='mass_rho_triple_prime', value=2.0000000002, is_fixed=True), Parameter(name='decay_rate_rho_triple_prime', value=0.2138095751275945, is_fixed=True)])

    results = calculate_deviations_from_fit(
        parameters,
        neutral_ts, neutral_cross_sections_values, neutral_errors, False,
        kaon_mass, alpha, hc_squared,
    )
    print('{0:^7s}    {1:^7s}    {2:^10s}    {3:^10s}    {4:^4s}'.format('t', 'error', 'measured', 'fit', 'deviation'))
    for r in results:
        print('{0:>7.4f}    {1:1.1e}    {2:.4e}    {3:.4e}    {4:>4.1f}'.format(*r))

    filename = '../neutral_data_errors_2.txt'
    with open(filename, 'w') as f:
        f.write('{0:^7s}    {1:^7s}    {2:^10s}    {3:^10s}    {4:^4s}\n'.format(
            't', 'error', 'measured', 'fit', 'deviation')
        )
        for t, err, cs, fit, dev in results:
            f.write('{0:>7.4f}    {1:1.1e}    {2:.4e}    {3:.4e}    {4:>4.1f}\n'.format(t, err, cs, fit, dev))
