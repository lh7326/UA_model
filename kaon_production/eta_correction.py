"""
This module implements the "eta" correction that is used to account
for final state radiation in the cross-section measurements.

See:
A. Hoefer, J. Gluza, F. Jegerlehner: Pion Pair Production with Higher Order Radiative Corrections in
Low Energy e^+ e^− Collisions, 2001, hep-ph/0107154

K. Melnikov: ON THE THEORETICAL UNCERTAINTIES IN THE MUON ANOMALOUS MAGNETIC MOMENT, 2001, hep-ph/0105267

"""
import math
import matplotlib.pyplot as plt
import scipy.special


def calculate_beta(s: float, final_particle_mass: float) -> float:
    """
    Returns the magnitude of the velocity of either of the two particles
    with the total four-momentum square equal to s in their center of mass system.

    That is v = p / E = sqrt((E^2 - m^2) / E^2) = sqrt((s - 4m^2) / s),
    because in CMS we have s = (2E)^2.

    Args:
        s (float):  total four-momentum squared
        final_particle_mass (float):  mass of either of the two particles (of the same mass)

    Returns: CMS velocity of either of the particles

    """
    return math.sqrt(1.0 - 4 * final_particle_mass**2 / s)


def dilogarithm(x: float) -> float:
    """
    Return the dilogarithm (or Spence's function) of the argument.

    There are two common conventions regarding this function.
    We use the definition
      Li_2(z) = - \int_0^z du \frac{\ln(1 - u)}{u};
    that is, our dilogarithm has a branch point at z = 1.

    Note: In this implementation the argument is restricted to real numbers
    in the interval (-infty, 1.0].

    Args:
        x (float): the argument of the Spence's function (restricted to x <= 1.0)

    Returns: float

    """
    return scipy.special.spence(1 - x)


def _calculate_eta(beta: float) -> float:
    intermediate_result = (
            4 * dilogarithm((1 - beta) / (1 + beta)) +
            2 * dilogarithm(-(1 - beta) / (1 + beta)) +
            (-3) * math.log(2 / (1 + beta)) * math.log((1 + beta) / (1 - beta)) +
            (-2) * math.log(beta) * math.log((1 + beta) / (1 - beta))
    )
    intermediate_result *= (1 + beta**2) / beta
    intermediate_result = intermediate_result - 3 * math.log(4.0 / (1 - beta**2)) - 4 * math.log(beta)
    intermediate_result += (1 / beta**3) * (1.25 * ((1 + beta**2)**2) - 2) * math.log((1 + beta) / (1 - beta))
    return intermediate_result + 1.5 * (1 + beta**2) / (beta**2)


def add_fsr_effects(
        cs_value: float,
        s: float,
        final_particle_mass: float,
        alpha: float
) -> float:
    """
    Returns a cross-section value with final state radiation effects added to it.
    Can be also applied on squared form factors.

    Args:
        cs_value (float): a cross-section value
        s (float): Mandelstam s-variable
        final_particle_mass (float): the mass of either of the two final particles
        alpha (float): the value of the fine structure constant

    Returns: a corrected cross-section value (float)

    """
    beta = calculate_beta(s, final_particle_mass)
    eta = _calculate_eta(beta)
    return cs_value * (1.0 + eta * alpha / math.pi)


def remove_fsr_effects(
        cs_value: float,
        s: float,
        final_particle_mass: float,
        alpha: float
) -> float:
    """
    Returns a cross-section value with final state radiation effects removed from it.
    Can be also applied on squared form factors.

    Args:
        cs_value (float): a cross-section value
        s (float): Mandelstam s-variable
        final_particle_mass (float): the mass of either of the two final particles
        alpha (float): the value of the fine structure constant

    Returns: a corrected cross-section value (float)

    """
    beta = calculate_beta(s, final_particle_mass)
    eta = _calculate_eta(beta)
    return cs_value / (1.0 + eta * alpha / math.pi)


if __name__ == '__main__':
    betas = [x / 100.0 for x in range(1, 100)]
    etas = [_calculate_eta(beta) for beta in betas]
    _, ax = plt.subplots()
    ax.set_title('Eta correction')
    ax.set_xlabel('Beta [1]')
    ax.set_ylabel('Eta [1]')
    ax.scatter(betas, etas)
    plt.show()
    plt.close()

    ss = [x / 100.0 for x in range(10, 100)]
    corrections = [add_fsr_effects(1.0, s, 0.13957039, 0.0072973525693)
                   for s in ss]
    _, ax = plt.subplots()
    ax.set_title('FSR correction')
    ax.set_xlabel('s [GeV^2]')
    ax.set_ylabel('FSR correction [1]')
    ax.scatter(ss, corrections)
    plt.show()
    plt.close()

    sqrt_s_list = [x / 100.0 for x in range(40, 200)]
    beta_list = [calculate_beta(sqrt_s**2, 0.13957039) for sqrt_s in sqrt_s_list]
    eta_list = [_calculate_eta(beta) for beta in beta_list]
    _, ax = plt.subplots()
    ax.set_title('Eta correction')
    ax.set_xlabel('sqrt(s) [GeV]')
    ax.set_ylabel('Eta [1]')
    ax.scatter(sqrt_s_list, eta_list)
    plt.show()
    plt.close()
