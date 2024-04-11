from configparser import ConfigParser
import math
from typing import Tuple, Union
from model_parameters import KaonParametersPhiRatio, KaonParametersPhiRatioSimple
from kaon_production.eta_correction import calculate_cremmer_gourdin_factor
from common.utils import make_partial_ff_or_cs_for_parameters
from kaon_production.data import KaonDatapoint


def _calculate_kinematic_factor(phi_mass: float, charged_kaon_mass: float, neutral_kaon_mass: float) -> float:
    def _beta(final_particle_mass):
        return math.sqrt(1.0 - 4 * final_particle_mass**2 / phi_mass**2)
    return (_beta(charged_kaon_mass) / _beta(neutral_kaon_mass))**3


def _calculate_kinematic_factor_with_errors(
        phi_mass: float, phi_mass_error: float, charged_kaon_mass: float, neutral_kaon_mass: float
) -> Tuple[float, float]:
    def _beta(final_particle_mass):
        value = math.sqrt(1.0 - 4 * final_particle_mass**2 / phi_mass**2)
        error = (4.0 / value) * (final_particle_mass**2 / phi_mass**3) * phi_mass_error
        return value, error
    num_val, num_err = _beta(charged_kaon_mass)
    den_val, den_err = _beta(neutral_kaon_mass)
    ratio = num_val / den_val
    ratio_err = abs(num_err / den_val - num_val * den_err / (den_val ** 2))
    return ratio**3, 3 * (ratio**2) * ratio_err


def calculate_r_ratio(
        parameters: Union[KaonParametersPhiRatio, KaonParametersPhiRatioSimple],
        charged_kaon_mass: float,
        neutral_kaon_mass: float,
        alpha: float,
        use_radiative_correction: bool = True,
) -> float:
    mass_phi = parameters['mass_phi'].value
    if use_radiative_correction:
        radiative_correction = calculate_cremmer_gourdin_factor(mass_phi**2, alpha, charged_kaon_mass)
    else:
        radiative_correction = 1
    coupling_constants_ratio = parameters['a_phi_charged'].value / parameters['a_phi_neutral'].value
    kinematic_factor = _calculate_kinematic_factor(mass_phi, charged_kaon_mass, neutral_kaon_mass)
    return radiative_correction * kinematic_factor * (coupling_constants_ratio**2)


def calculate_cross_sections_ratio_at_phi_peak(
        parameters: Union[KaonParametersPhiRatio, KaonParametersPhiRatioSimple],
        charged_kaon_mass: float,
        neutral_kaon_mass: float,
        alpha: float,
        hc_squared: float,
        use_radiative_correction: bool = True,
) -> float:
    parameters = parameters.copy()
    mass_phi = parameters['mass_phi'].value
    if use_radiative_correction:
        radiative_correction = calculate_cremmer_gourdin_factor(mass_phi**2, alpha, charged_kaon_mass)
    else:
        radiative_correction = 1
        parameters.fix_all_parameters()
    f = make_partial_ff_or_cs_for_parameters(
            alpha, hc_squared, parameters, charged_kaon_mass, neutral_kaon_mass
    )
    phi_mass = parameters['mass_phi'].value
    ts = [
        KaonDatapoint(t=phi_mass**2, is_charged=True, is_for_cross_section=True),
        KaonDatapoint(t=phi_mass**2, is_charged=False, is_for_cross_section=True),
    ]
    res = f(ts)

    return radiative_correction * res[0] / res[1]


def calculate_r_ratio_with_error(
        parameters: Union[KaonParametersPhiRatio, KaonParametersPhiRatioSimple],
        charged_kaon_mass: float,
        neutral_kaon_mass: float,
        alpha: float,
        mass_phi_error: float,
        a_phi_charged_error: float,
        a_phi_neutral_error: float,
        use_radiative_correction: bool = True,
) -> Tuple[float, float]:
    mass_phi = parameters['mass_phi'].value
    if use_radiative_correction:
        radiative_correction = calculate_cremmer_gourdin_factor(mass_phi**2, alpha, charged_kaon_mass)
    else:
        radiative_correction = 1
    a_phi_charged = parameters['a_phi_charged'].value
    a_phi_neutral = parameters['a_phi_neutral'].value
    coupling_constants_ratio = a_phi_charged / a_phi_neutral
    coupling_constants_ratio_error = (
            abs(a_phi_charged_error / a_phi_neutral) +
            abs(a_phi_charged * a_phi_neutral_error / a_phi_neutral**2)
    )
    kinematic_factor, kin_fac_err = _calculate_kinematic_factor_with_errors(
        mass_phi, mass_phi_error, charged_kaon_mass, neutral_kaon_mass)
    err1 = kin_fac_err * (coupling_constants_ratio**2)
    err2 = kinematic_factor * 2 * abs(coupling_constants_ratio) * coupling_constants_ratio_error
    err = radiative_correction * (err1 + err2)
    return radiative_correction * kinematic_factor * (coupling_constants_ratio**2), err


def read_error_from_report(report_filepath):
    from model_parameters import Parameter
    from numpy import array

    def find(pars, name):
        for i, p in enumerate(pars):
            if p.name == name:
                return i
        else:
            raise ValueError(f'Cannot find: {name}')

    def get_errors(data):
        errors_list = data['parameter_errors']
        final_parameters = data['final_parameters']
        free_parameters = [p for p in final_parameters if not p.is_fixed]
        return [errors_list[find(free_parameters, name)] for name
                in ['mass_phi', 'a_phi_charged', 'a_phi_neutral']]

    with open(report_filepath, 'r') as f:
        data = eval(f.read())

    return get_errors(data)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')
    hc_squared = config.getfloat('constants', 'hc_squared')
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    neutral_kaon_mass = config.getfloat('constants', 'neutral_kaon_mass')

    for i in [33, 51, 74, 76]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runPhiRatio1_{i}/final_fit_parameters.pickle'
        kaon_errors_filepath = f'/home/lukas/reports/kaons/runPhiRatio1_{i}/report_200.txt'
        kaon_parameters = KaonParametersPhiRatio.load_from_serialized_parameters(kaon_parameters_filepath)
        mass_phi_error, a_phi_charged_error, a_phi_neutral_error = read_error_from_report(kaon_errors_filepath)

        cs_ratio = calculate_cross_sections_ratio_at_phi_peak(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, False)

        r_ratio_no_rc = calculate_r_ratio(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, use_radiative_correction=False)
        r_ratio_no_rc2, err_no_rc = calculate_r_ratio_with_error(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha,
            mass_phi_error, a_phi_charged_error, a_phi_neutral_error, use_radiative_correction=False)
        assert r_ratio_no_rc == r_ratio_no_rc2
        r_ratio = calculate_r_ratio(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, use_radiative_correction=True)
        r_ratio2, err = calculate_r_ratio_with_error(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha,
            mass_phi_error, a_phi_charged_error, a_phi_neutral_error, use_radiative_correction=True)
        assert r_ratio == r_ratio2
        print(f'Parameters runPhiRatio1_{i}: cs ratio (no RC): {cs_ratio}  R ratio (no RC): {r_ratio_no_rc}, {err_no_rc} ; R ratio: {r_ratio}, {err}')

    for i in [13, 32, 41]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runPhiRatio2_{i}/final_fit_parameters.pickle'
        kaon_errors_filepath = f'/home/lukas/reports/kaons/runPhiRatio2_{i}/report_200.txt'
        kaon_parameters = KaonParametersPhiRatioSimple.load_from_serialized_parameters(kaon_parameters_filepath)
        mass_phi_error, a_phi_charged_error, a_phi_neutral_error = read_error_from_report(kaon_errors_filepath)

        cs_ratio = calculate_cross_sections_ratio_at_phi_peak(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, hc_squared, False)

        r_ratio_no_rc = calculate_r_ratio(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, use_radiative_correction=False)
        r_ratio_no_rc2, err_no_rc = calculate_r_ratio_with_error(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha,
            mass_phi_error, a_phi_charged_error, a_phi_neutral_error, use_radiative_correction=False)
        assert r_ratio_no_rc == r_ratio_no_rc2
        r_ratio = calculate_r_ratio(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha, use_radiative_correction=True)
        r_ratio2, err = calculate_r_ratio_with_error(
            kaon_parameters, charged_kaon_mass, neutral_kaon_mass, alpha,
            mass_phi_error, a_phi_charged_error, a_phi_neutral_error, use_radiative_correction=True)
        assert r_ratio == r_ratio2
        print(f'Parameters runPhiRatio2_{i}: cs ratio (no RC): {cs_ratio} R ratio (no RC): {r_ratio_no_rc}, {err_no_rc} ; R ratio: {r_ratio}, {err}')
