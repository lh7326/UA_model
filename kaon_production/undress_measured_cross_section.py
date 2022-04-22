from configparser import ConfigParser
import math


def error_for_addition(err1, err2):
    return math.sqrt(err1**2 + err2**2)


def error_for_multiplication(val1, err1, val2, err2):
    rel_err1 = err1 / val1
    rel_err2 = err2 / val2
    return val1 * val2 * error_for_addition(rel_err1, rel_err2)


def undress_cross_section_measurement(alpha_0, alpha_s, alpha_err, cs_val, cs_err_stat, cs_err_sys=None):
    if cs_err_sys is None:
        cs_err = cs_err_stat
    else:
        cs_err = error_for_addition(cs_err_stat, cs_err_sys)

    factor = (alpha_0 / alpha_s) ** 2
    factor_err = 2 * (factor / alpha_s) * alpha_err

    return cs_val * factor, error_for_multiplication(factor, factor_err, cs_val, cs_err)


if __name__ == '__main__':
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')

    e = 2.95
    alpha_s = 7.40270E-03
    alpha_err = 3.96599E-06
    cs = 0.0322
    cs_err_stat = 0.0027
    cs_err_sys = 0.0012
    print(e**2, undress_cross_section_measurement(alpha, alpha_s, alpha_err, cs, cs_err_stat, cs_err_sys))
