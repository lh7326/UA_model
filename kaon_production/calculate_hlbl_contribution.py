from configparser import ConfigParser
from typing import Callable, Tuple
from common.utils import make_partial_form_factor_for_parameters
from model_parameters import KaonParametersSimplified, Parameter, PionParameters
from kaon_production.data import KaonDatapoint
from ua_model.PionUAModel import PionUAModel

import math
import numpy
from scipy.integrate import dblquad, tplquad


def _wrap_partial_form_factor_function(partial_f: Callable, charged: bool = True) -> Callable:
    def wrapped(s):
        datapoint = KaonDatapoint(t=s, is_charged=charged, is_for_cross_section=False)
        res = partial_f([datapoint])[0].real
        return res
    return wrapped


def _make_partial_for_pion_parameters(pion_parameters: PionParameters) -> Callable:
    pion_parameters = pion_parameters.copy()
    ff_model = PionUAModel(**{p.name: p.value for p in pion_parameters})

    def wrapped(s):
        res = ff_model(s).real
        return res
    return wrapped


def _calculate_i_n(x, y, q1squared, q2squared, q3squared, n, particle_mass):
    m2 = particle_mass**2

    def _calculate_delta_2q(k1squared, k2squared):
        return m2 - x * (1 - x) * k1squared - y * (1 - y) * k2squared

    def _calculate_delta_3q(k1squared, k2_squared, k3_squared):
        return m2 - x * y * k1squared - x * (1 - x - y) * k2_squared - y * (1 - x - y) * k3_squared

    if n == 1:
        delta_123 = _calculate_delta_3q(q1squared, q2squared, q3squared)
        delta_23 = _calculate_delta_2q(q2squared, q3squared)
        return 8 * x * y * (1 - 2*x) * (1 - 2*y) / (delta_123 * delta_23)
    elif n == 4:
        delta_321 = _calculate_delta_3q(q3squared, q2squared, q1squared)
        delta_21 = _calculate_delta_2q(q2squared, q1squared)
        return (
            (4 * (1 - x - y) * (1 - 2*x - 2*y) * delta_21 / delta_321**2) *
            ((1 - 2*x - 2*y)**2 / delta_321 - (1 - x * (3 - 2*x) - y * (3 - 2*y)) / delta_21)
            + 16 * x * y * (1 - 2*x) * (1 - 2*y) / (delta_321 * delta_21))
    elif n == 7:
        delta_123 = _calculate_delta_3q(q1squared, q2squared, q3squared)
        return -8 * x * y * (1 - x - y) * ((1 - 2*x)**2) * (1 - 2*y) / (delta_123**3)
    elif n == 17:
        delta_123 = _calculate_delta_3q(q1squared, q2squared, q3squared)
        delta_23 = _calculate_delta_2q(q2squared, q3squared)
        return 16 * x * (y**2) * (1 - 2*x) * (1 - 2*y) * (
            (1 - x - y) / delta_123 + (1 - y) / delta_23
        ) / (delta_123 * delta_23)
    elif n == 39:
        delta_123 = _calculate_delta_3q(q1squared, q2squared, q3squared)
        return 8 * x * y * (1 - x - y) * (1 - 2*x) * (1 - 2*y) * (1 - 2*x - 2*y) / (delta_123**3)
    elif n == 54:
        delta_321 = _calculate_delta_3q(q3squared, q2squared, q1squared)
        delta_21 = _calculate_delta_2q(q2squared, q1squared)
        return -8 * x * y * (1 - x - y) * (1 - 2*x) * (1 - 2*y) * (x - y) * (
            1 / delta_321 + 1 / delta_21
        ) / (delta_321 * delta_21)
    else:
        raise NotImplementedError


def _calculate_pi_n(form_factor_function, q1squared, q2squared, q3squared, n, particle_mass):
    if n not in {1, 4, 7, 17, 39, 54}:
        raise NotImplementedError
    integral = dblquad(
        lambda y, x: _calculate_i_n(x, y, q1squared, q2squared, q3squared, n, particle_mass),
        a=0, b=1, gfun=0, hfun=lambda x: 1-x, epsrel=1.49e-3
    )[0]
    return (
        form_factor_function(q1squared) * form_factor_function(q2squared) *
        form_factor_function(q3squared) * integral / (16 * (math.pi**2))
    )


def calculate_pi_n(form_factor_function, q1squared, q2squared, q3squared, n, particle_mass):
    # we pass from Euclidean to Minkowskian quantities
    q1squared = -q1squared
    q2squared = -q2squared
    q3squared = -q3squared
    if n in {1, 4, 7, 17, 39, 54}:
        return _calculate_pi_n(form_factor_function, q1squared, q2squared, q3squared, n, particle_mass)
    elif n == 2:
        return _calculate_pi_n(form_factor_function, q1squared, q3squared, q2squared, 1, particle_mass)
    elif n == 5:
        return _calculate_pi_n(form_factor_function, q1squared, q3squared, q2squared, 4, particle_mass)
    elif n == 9:
        return _calculate_pi_n(form_factor_function, q3squared, q1squared, q2squared, 7, particle_mass)
    elif n == 10:
        return _calculate_pi_n(form_factor_function, q1squared, q3squared, q2squared, 7, particle_mass)
    elif n == 11:
        return _calculate_pi_n(form_factor_function, q3squared, q2squared, q1squared, 17, particle_mass)
    elif n == 50:
        return -1 * _calculate_pi_n(form_factor_function, q1squared, q3squared, q2squared, 54, particle_mass)
    else:
        ValueError(f'Unexpected value of n={n}!')


def _transform_variables(sigma: float, r: float, phi: float) -> Tuple[float, float, float]:
    """
    Calculate variables q1squared, q2squared, q3squared from sigma, r, phi.

    Returns:
        q1squared, q2squared, q3squared

    """
    c = sigma / 3
    a = 1 - 0.5 * r * math.cos(phi)
    b = 0.5 * r * math.sqrt(3) * math.sin(phi)
    return c * (a - b), c * (a + b), c * (1 + r * math.cos(phi))


def _calculate_sigma(q_squared: float, muon_mass_squared: float) -> float:
    return math.sqrt(1.0 + 4.0 * muon_mass_squared / q_squared)


def _calculate_tau(q1squared: float, q2squared: float, q3squared: float) -> float:
    return (
        (q3squared - q1squared - q2squared)
        / (2.0 * math.sqrt(q1squared) * math.sqrt(q2squared))
    )


def _calculate_capital_x(q1squared: float, q2squared: float, tau: float, muon_mass_squared: float) -> float:
    x = math.sqrt(1 - tau**2)
    q1 = math.sqrt(q1squared)
    q2 = math.sqrt(q2squared)
    z = (q1 * q2 * (1 - _calculate_sigma(q1squared, muon_mass_squared))
         * (1 - _calculate_sigma(q2squared, muon_mass_squared))) / (4 * muon_mass_squared)
    return math.atan(z * x / (1.0 - z * tau)) / (q1 * q2 * x)


def calculate_t_n(q1squared, q2squared, q3squared, n, muon_mass):
    muon_mass_squared = muon_mass**2
    sigma1 = _calculate_sigma(q1squared, muon_mass_squared)
    sigma2 = _calculate_sigma(q2squared, muon_mass_squared)
    q1 = math.sqrt(q1squared)
    q2 = math.sqrt(q2squared)
    tau = _calculate_tau(q1squared, q2squared, q3squared)
    if n == 1:
        q1q2 = q1 * q2
        num = q1squared * tau * (sigma1 - 1.0) * (sigma1 + 5.0)
        num += q2squared * tau * (sigma2 - 1.0) * (sigma2 + 5.0)
        num += 4.0 * q1q2 * (sigma1 + sigma2 - 2.0)
        num -= 8 * tau * muon_mass_squared
        return num / (2 * q1q2 * q3squared * muon_mass_squared)
    elif n == 2:
        a = q1 * (sigma1 - 1.0) * (q1 * tau * (sigma1 + 1.0) + 4.0 * q2 * (tau**2 - 1)) - 4 * tau * muon_mass_squared
        a = a / (q1 * q2 * q3squared * muon_mass_squared)
        b = _calculate_capital_x(
            q1squared, q2squared, tau, muon_mass_squared
        ) * 8.0 * (tau**2 - 1.0) * (2 * muon_mass_squared - q2squared)
        b = b / (q3squared * muon_mass_squared)
        return a + b
    elif n == 4:
        a = -2.0 * (sigma1 + sigma2 - 2.0) / muon_mass_squared
        b = -1.0 * q1 * tau * (sigma1 - 1.0) * (sigma1 + 7.0) / (2.0 * q2 * muon_mass_squared)
        c = -1.0 * q2 * tau * (sigma2 - 1.0) * (sigma2 + 7.0) / (2.0 * q1 * muon_mass_squared)
        d = 8.0 * tau / (q1 * q2)
        e = q1squared * (1.0 - sigma1) / (q2squared * muon_mass_squared)
        f = q2squared * (1.0 - sigma2) / (q1squared * muon_mass_squared)
        first_part = (1.0 / q3squared) * (a + b + c + d + e + f + 2.0 / q1squared + 2.0 / q2squared)
        second_part = (4.0 / muon_mass_squared - 8.0 * tau / (q1 * q2)) * _calculate_capital_x(
            q1squared, q2squared, tau, muon_mass_squared)
        return first_part + second_part
    elif n == 5:
        a = 4.0 * ((tau**2) * (sigma1 - 1.0) + sigma2 - 1.0) / muon_mass_squared
        b = -1.0 * q1 * tau * (sigma1 - 5.0) * (sigma1 - 1.0) / (q2 * muon_mass_squared)
        c = 4.0 * tau / (q1 * q2)
        d = -1.0 * q2 * tau * (sigma2 - 3.0) * (sigma2 - 1.0) / (q1 * muon_mass_squared)
        e = 2.0 * q2squared * (sigma2 - 1.0) / (q1squared * muon_mass_squared)
        f = -4.0 / q1squared
        g1 = (-8.0 * q2squared * (tau**2) - 16.0 * q1 * q2 * tau - 8.0 * q1squared) / muon_mass_squared
        g2 = (16.0 * tau * q2 / q1) + 16.0
        g = (g1 + g2) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return (a + b + c + d + e + f + g) / q3squared
    elif n == 7:
        a = q1squared * (
            ((tau**2) * (sigma1 - 1.0) * (sigma1 + 3.0) + 4.0 * (sigma1 + sigma2 - 2.0))/(2 * muon_mass_squared)
            - 4.0/q2squared
        )
        b = -1.0 * q2squared * (tau**2) * (sigma2 - 5.0) * (sigma2 - 1.0) / (2 * muon_mass_squared)
        c = q1 * q1squared * tau * (sigma1 - 1.0) * (sigma1 + 5.0) / (q2 * muon_mass_squared)
        d = q1 * (q2 * tau * (sigma1 + 5.0 * sigma2 - 6.0) / muon_mass_squared - 12.0 * tau / q2)
        e = 2.0 * q1squared * q1squared * (sigma1 - 1.0) / (q2squared * muon_mass_squared) - 4.0 * (tau**2)
        f0 = q1 * (8.0 * q2 * (tau**3 + tau) - 2.0 * q2 * q2squared * tau / muon_mass_squared)
        f1 = q1squared * (32.0 * (tau**2) - 4.0 * q2squared * (tau**2 + 1) / muon_mass_squared)
        f2 = q1 * q1squared * (16.0 * tau / q2 - 10.0 * q2 * tau / muon_mass_squared)
        f3 = -4.0 * q1squared * q1squared / muon_mass_squared
        f = (f0 + f1 + f2 + f3) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return (a + b + c + d + e + f) / q3squared
    elif n == 9:
        a = q1squared * (
                (tau**2) * ((sigma1 - 22.0) * sigma1 - 8.0 * sigma2 + 29.0)
                + 2.0 * (-5.0 * sigma1 + sigma2 + 4.0)
        ) / (2 * muon_mass_squared)
        b = q1 * (
            q2 * tau * (
                2.0 * (tau**2) * ((sigma2 - 3.0)**2 - 4.0 * sigma1)
                - 26.0 * sigma1 + sigma2 * (sigma2 - 12.0) + 37.0
        ) / (2 * muon_mass_squared) - 4.0 * tau /q2
        )
        c = q2squared * (
                (tau**2) * (-8.0 * sigma1 + sigma2 * (5.0 * sigma2 - 26.0) + 29.0)
                - 4.0 * (sigma1 + 2.0 * sigma2 - 3.0)
        ) / (2 * muon_mass_squared)
        d = q1 * q1squared * tau * (sigma1 - 9.0) * (sigma1 - 1.0) / (2.0 * q2 * muon_mass_squared)
        e = q2 * q2squared * tau * (sigma2 - 9.0) * (sigma2 - 1.0) / (q1 * muon_mass_squared)
        f = (8.0 * q2 * tau / q1 +
             2.0 * q2squared * q2squared * (1 - sigma2) / (q1squared * muon_mass_squared) +
             4.0 * q2squared / q1squared)
        g0 = q2 * q1 * q1squared * tau * (8.0 * tau**2 + 22.0) / muon_mass_squared
        g1 = q1squared * q1squared * (8.0 * tau**2 - 2.0) / muon_mass_squared
        g2 = q1squared * (
            q2squared * (36.0 * tau**2 + 18.0) / muon_mass_squared
            - 8.0 * (tau**2 + 1.0)
        )
        g3 = q2squared * q2squared * (8.0 * tau**2 + 4.0) / muon_mass_squared
        g4 = q1 * (
            q2 * q2squared * (8.0 * tau**3 + 34.0 * tau) / muon_mass_squared
            - 8.0 * q2 * tau * (tau**2 + 5.0)
        )
        g5 = -16.0 * q2squared * ((2.0 * tau**2 + 1.0) + tau * q2 / q1)
        g = (g0 + g1 + g2 + g3 + g4 + g5) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return (a + b + c + d + e + f + g) / q3squared
    elif n == 10:
        a = q1squared * (
            2.0 * (sigma1 + sigma2 - 2.0)
            - (tau**2) * ((sigma1 + 10.0) * sigma1 + 8.0 * sigma2 - 19.0)
        ) / (2.0 * muon_mass_squared)
        b = q1 * (
            q2 * tau * (2.0 * tau**2 * (sigma2 - 5.0) * (sigma2 - 1.0) - 2.0 * sigma1
                        + sigma2 * (sigma2 + 4.0) - 3.0) / (2.0 * muon_mass_squared)
            - 4.0 * tau / q2
        )
        c = q2squared * (tau**2) * (sigma2 - 5.0) * (sigma2 - 1.0) / (2.0 * muon_mass_squared)
        d = q1 * q1squared * tau * (sigma1 - 9.0) * (sigma1 - 1.0) / (2.0 * q2 * muon_mass_squared)
        e = 4.0 * (tau ** 2)
        f0 = q2 * q1 * q1squared * tau * (8.0 * tau**2 + 6.0) / muon_mass_squared
        f1 = q1 * (2.0 * q2 * q2squared * tau / muon_mass_squared - 8.0 * q2 * (tau**3 + tau))
        f2 = q1squared * q1squared * (8.0 * tau**2 - 2.0) / muon_mass_squared
        f3 = q1squared * (
            2.0 * q2squared * (6.0 * (tau**2) - 1) / muon_mass_squared
            - 8.0 * (tau**2 + 1)
        )
        f = (f0 + f1 + f2 + f3) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return (a + b + c + d + e + f) / q3squared
    elif n == 11:
        a = q1squared * (
            4.0 / q2squared - 2.0 * (2.0 * (tau**2) + 1.0) * (sigma1 + sigma2 - 2.0) / muon_mass_squared
        )
        b = q1 * (
            4.0 * tau / q2 - 4.0 * q2 * tau * (tau**2 + 1.0) * (sigma2 - 1.0) / muon_mass_squared
        )
        c = -6.0 * q1 * q1squared * tau * (sigma1 - 1.0) / (q2 * muon_mass_squared)
        d = q1squared * q1squared * (2.0 - 2.0 * sigma1) / (q2squared * muon_mass_squared)
        e0 = q1squared * q1squared * (8.0 * (tau**2) + 4.0) / muon_mass_squared
        e1 = q1 * q1squared * (
            8.0 * q2 * tau * (tau**2 + 2.0) / muon_mass_squared
            - 16.0 * tau / q2
        )
        e2 = q1squared * (q2squared * (8.0 * (tau**2) + 4.0) / muon_mass_squared - 16.0 * tau**2)
        e = (e0 + e1 + e2) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return (a + b + c + d + e) / q3squared
    elif n == 17:
        a = q3squared * (
            (sigma1 - 1.0) / (q2squared * muon_mass_squared) +
            (sigma2 - 1.0) / (q1squared * muon_mass_squared) -
            2.0 / (q1squared * q2squared)
        )
        b0 = -2.0 * q3squared / muon_mass_squared
        b1 = 8.0 * q2 * tau / q1  + 8.0 * q1 * tau / q2
        b2 = 8.0 * (tau**2 + 1.0)
        b = (b0 + b1 + b2) * _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        return a + b
    elif n == 39:
        capital_x = _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        a = -1.0 * q1squared * (
            (tau**2) * (sigma1 - 1.0) * (sigma1 + 3.0) + 2.0 * (sigma1 + sigma2 - 2.0)
        ) / muon_mass_squared
        b = -1.0 * q2 * q2squared * tau * (sigma2 - 1.0) * (sigma2 + 3.0) / (q1 * muon_mass_squared)
        c = -1.0 * q2squared * (
            (tau**2) * (sigma2 - 1.0) * (sigma2 + 3.0) + 2.0 * (sigma1 + sigma2 - 2.0)
        ) / muon_mass_squared
        d = -1.0 * q1 * q1squared * tau * (sigma1 - 1.0) * (sigma1 + 3.0) / (q2 * muon_mass_squared)
        e = q1 * (
            8.0 * tau / q2 -
            q2 * tau * ((sigma1 + 4.0) * sigma1 + (sigma2 + 4.0) * sigma2 - 10.0) / muon_mass_squared
        )
        f = 8.0 * q2 * tau / q1 + 8.0 * (tau ** 2)
        g = capital_x * (-16.0 * ((tau**2) - 1.0) * (q1squared + q1 * q2 * tau + q2squared))
        return (a + b + c + d + e + f + g) / (2.0 * q3squared) + (
                2.0 * (q1 * q2 * tau + q1squared + q2squared) / muon_mass_squared) * capital_x
    elif n == 50:
        capital_x = _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        a = q2squared * q2squared * q2 * tau * (-6.0 * sigma2 + sigma2**2 + 5.0)
        b = 8.0 * q1squared * q1squared * q1 * (
            -1.0 * sigma1 + 2.0 * q2squared * (tau**2 + 1.0) * capital_x + 1.0
        )
        c = 4.0 * q2 * q1squared * q1squared * tau * (
            -7.0 * sigma1 + 2.0 * q2squared * (2.0 * (tau**2) + 9.0) * capital_x + 7.0
        )
        d = 4.0 * q2squared * q1squared * q1 * (
            2.0 * (tau**2) * (-3.0 * sigma1 - sigma2 + 8.0 * q2squared * capital_x + 4.0)
            - 2.0 * (sigma1 + sigma2 - 2.0) + 5.0 * q2squared * capital_x
        )
        e = q2squared * q2 * q1squared * tau * (
            8.0 * (tau**2) * (-1.0 * sigma1 - sigma2 + 2.0 * q2squared * capital_x + 2.0)
            - 6.0 * sigma1 - sigma1**2 - 28.0 * sigma2 + 16.0 * q2squared * capital_x + 35.0
        )
        f = 2.0 * q2squared * q2squared * q1 * (
            (tau**2) * (-10.0 * sigma2 + sigma2**2 + 9.0) - sigma1 - 3.0 * sigma2
            + 2.0 * q2squared * capital_x + 4.0
        )
        g0 = -1.0 * q2 * q2squared * tau
        g1 = 2.0 * q1 * q1squared * (2.0 * q2squared * (4.0 * (tau**2) + 1.0) * capital_x - 1.0)
        g2 = q2 * q1squared * tau * (4.0 * q2squared * (tau**2 + 3.0) * capital_x - 5.0)
        g3 = q2squared * q1 * (
                2.0 * (tau**2) * (q2squared * capital_x - 1.0) + 2.0 * q2squared * capital_x - 1.0)
        g4 = 8.0 * q2 * q1squared * q1squared * tau * capital_x
        g = -8.0 * muon_mass_squared * (g0 + g1 + g2 + g3 + g4)
        return (a + b + c + d + e + f + g) / (2.0 * muon_mass_squared * q1 * q2squared * q3squared)
    elif n == 54:
        capital_x = _calculate_capital_x(q1squared, q2squared, tau, muon_mass_squared)
        a = q2squared * tau * (
                -1.0 * q3squared * sigma2**2 + q2squared * (6.0 * sigma2 - 5.0) - 8.0 * muon_mass_squared)
        b = -2.0 * q2 * q1 * q1squared * (
            (tau ** 2) * (2.0 * sigma1 + 8.0 * capital_x * muon_mass_squared - 1.0)
            - 3.0 * sigma1 + sigma2 + 8.0 * capital_x * muon_mass_squared + 2.0
        )
        c = q1squared * tau * (
            -2.0 * q2squared * (4.0 * (tau ** 2) - 5.0) * (sigma1 - sigma2)
            + q3squared * (sigma1**2) + 8.0 * muon_mass_squared
            + 8.0 * q2squared * q2squared * (2.0 * (tau ** 2) - 3.0) * capital_x
        )
        d = 2.0 * q2squared * q2 * q1 * (
            (tau ** 2) * (2.0 * sigma2 + 8.0 * capital_x * muon_mass_squared - 1.0)
            + sigma1 - 3.0 * sigma2 + 8.0 * capital_x * muon_mass_squared
            - 2.0 * q2squared * capital_x + 2.0
        )
        e = q1squared * q1squared * tau * (
            -6.0 * sigma1 - 8.0 * q2squared * (2.0 * (tau ** 2) - 3.0) * capital_x + 5.0
        )
        f = 4.0 * q2 * q1squared * q1squared * q1 * capital_x
        return (a + b + c + d + e + f) / (4.0 * muon_mass_squared * q1 * q2 * q3squared)
    else:
        raise ValueError(f'Unsupported value of n: {n}')


def calculate_hlbl_contribution(form_factor_function, alpha, particle_mass, muon_mass):
    def integrand(phi, r, sigma):
        q1squared, q2squared, q3squared = _transform_variables(sigma, r, phi)
        acc = 0
        for n in [1, 2, 4, 5, 7, 9, 10, 11, 17, 39, 50, 54]:
            kernel = calculate_t_n(q1squared, q2squared, q3squared, n, muon_mass)
            pi = calculate_pi_n(form_factor_function, q1squared, q2squared, q3squared, n, particle_mass)
            acc += kernel * pi
        return acc
    return (alpha**3 / (432 * (math.pi**2)) ) * tplquad(
        integrand, a=0, b=numpy.inf, gfun=0, hfun=1, qfun=0, rfun=2*math.pi)[0]


if __name__ == '__main__':

    config = ConfigParser(inline_comment_prefixes='#')
    config.read('../configuration.ini')
    alpha = config.getfloat('constants', 'alpha')
    charged_kaon_mass = config.getfloat('constants', 'charged_kaon_mass')
    muon_mass = config.getfloat('constants', 'muon_mass')

    for i in [2, 29, 40, 125, 151]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/run9simplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=True
        )
        em_mass2_charged_kaon = calculate_hlbl_contribution(f, alpha, charged_kaon_mass, muon_mass)
        print(f'{kaon_parameters_filepath} HLbL contribution charged kaon: {em_mass2_charged_kaon}')

        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=False
        )
        em_mass2_neutral_kaon = calculate_hlbl_contribution(f, alpha, charged_kaon_mass, muon_mass)
        print(f'{kaon_parameters_filepath} HLbL neutral kaon: {em_mass2_neutral_kaon}')

    for i in [37, 113, 160, 165]:
        kaon_parameters_filepath = f'/home/lukas/reports/kaons/runTestDressedsimplified_{i}/final_fit_parameters.pickle'
        kaon_parameters = KaonParametersSimplified.load_from_serialized_parameters(kaon_parameters_filepath)
        kaon_parameters.fix_all_parameters()
        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=True
        )
        em_mass2_charged_kaon = calculate_hlbl_contribution(f, alpha, charged_kaon_mass, muon_mass)
        print(f'{kaon_parameters_filepath} HLbL contribution charged kaon: {em_mass2_charged_kaon}')

        f = _wrap_partial_form_factor_function(
            make_partial_form_factor_for_parameters(kaon_parameters, return_absolute_value=False),
            charged=False
        )
        em_mass2_neutral_kaon = calculate_hlbl_contribution(f, alpha, charged_kaon_mass, muon_mass)
        print(f'{kaon_parameters_filepath} HLbL contribution neutral kaon: {em_mass2_neutral_kaon}')
