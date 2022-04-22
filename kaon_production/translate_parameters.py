from configparser import ConfigParser
from kaon_production.ModelParameters import ModelParameters, Parameter

if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2

    model_parameters = ModelParameters.from_list([Parameter(name='t_0_isoscalar', value=0.17531904388276887, is_fixed=True), Parameter(name='t_0_isovector', value=0.07791957505900839, is_fixed=True), Parameter(name='t_in_isoscalar', value=1.719520190300892, is_fixed=False), Parameter(name='t_in_isovector', value=1.3314211338413366, is_fixed=False), Parameter(name='a_omega', value=0.9028911434086133, is_fixed=False), Parameter(name='mass_omega', value=0.7750000000000052, is_fixed=False), Parameter(name='decay_rate_omega', value=0.008899911535638454, is_fixed=False), Parameter(name='a_omega_prime', value=0.03221862128992986, is_fixed=False), Parameter(name='mass_omega_prime', value=1.3875452756787783, is_fixed=False), Parameter(name='decay_rate_omega_prime', value=0.25000000000000006, is_fixed=False), Parameter(name='a_omega_double_prime', value=0.2308606519667505, is_fixed=False), Parameter(name='mass_omega_double_prime', value=1.6749999999999998, is_fixed=False), Parameter(name='decay_rate_omega_double_prime', value=0.32999999999999996, is_fixed=False), Parameter(name='a_phi', value=-0.3265301513590924, is_fixed=False), Parameter(name='mass_phi', value=1.0197565568295128, is_fixed=False), Parameter(name='decay_rate_phi', value=0.004499999999999999, is_fixed=False), Parameter(name='a_phi_prime', value=-0.015805307658535144, is_fixed=False), Parameter(name='mass_phi_prime', value=1.6750000000000003, is_fixed=False), Parameter(name='decay_rate_phi_prime', value=0.10000000000000002, is_fixed=False), Parameter(name='mass_phi_double_prime', value=2.1854329339382543, is_fixed=False), Parameter(name='decay_rate_phi_double_prime', value=0.10923779542006498, is_fixed=False), Parameter(name='a_rho', value=-0.39529949950263177, is_fixed=False), Parameter(name='mass_rho', value=0.7799999999994054, is_fixed=False), Parameter(name='decay_rate_rho', value=0.14999999999999997, is_fixed=False), Parameter(name='a_rho_prime', value=0.7071726765500697, is_fixed=False), Parameter(name='mass_rho_prime', value=1.2884577879671701, is_fixed=False), Parameter(name='decay_rate_rho_prime', value=0.5499999999999999, is_fixed=False), Parameter(name='a_rho_double_prime', value=-0.13496289837794273, is_fixed=False), Parameter(name='mass_rho_double_prime', value=1.7399999999999998, is_fixed=False), Parameter(name='decay_rate_rho_double_prime', value=0.22799305025607813, is_fixed=False), Parameter(name='mass_rho_triple_prime', value=2.18616762017487, is_fixed=False), Parameter(name='decay_rate_rho_triple_prime', value=0.10891038421314707, is_fixed=False)])
    print(model_parameters.get_ordered_values())

    # model_parameters = ModelParameters.from_ordered_values([
    #     0.7908450,
    #     1.3833681,
    #     0.75823,
    #     1.34231,
    #     1.7182,
    #     2.3053731,
    #     0.14456,
    #     0.49217,
    #     0.48958,
    #     0.5768760,
    #     -0.1377353,
    #     0.0389892,
    #     0.5898944,
    #     0.78266,
    #     1.019461,
    #     1.2261584,
    #     1.7422655,
    #     1.8109782,
    #     2.2332353,
    #     0.00868,
    #     0.004249,
    #     0.7375916,
    #     0.2935377,
    #     0.8568563,
    #     0.0522538,
    #     0.3139914,
    #     -0.3325257,
    #     1.0696948,
    #     -0.2672411,
    #     -0.2859869,
    # ], t_0_isoscalar=0.17531904388276887, t_0_isovector=0.07791957505900839,
    # )
    # print(model_parameters.to_list())