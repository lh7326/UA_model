from configparser import ConfigParser
from kaon_production.function import function_form_factor
from kaon_production.ModelParameters import ModelParameters


def make_partial_for_parameters(parameters: ModelParameters):
    def _from_parameters_or_arguments(name, args):
        parameter = parameters[name]
        if parameter.is_fixed:
            return parameter.value
        else:
            return args.pop()

    def partial_f(ts, *args):
        args = list(args)

        # WARNING: the order of the commands below is important!
        decay_rate_rho_triple_prime = _from_parameters_or_arguments('decay_rate_rho_triple_prime', args)
        mass_rho_triple_prime = _from_parameters_or_arguments('mass_rho_triple_prime', args)
        decay_rate_rho_double_prime = _from_parameters_or_arguments('decay_rate_rho_double_prime', args)
        mass_rho_double_prime = _from_parameters_or_arguments('mass_rho_double_prime', args)
        a_rho_double_prime = _from_parameters_or_arguments('a_rho_double_prime', args)
        decay_rate_rho_prime = _from_parameters_or_arguments('decay_rate_rho_prime', args)
        mass_rho_prime = _from_parameters_or_arguments('mass_rho_prime', args)
        a_rho_prime = _from_parameters_or_arguments('a_rho_prime', args)
        decay_rate_rho = _from_parameters_or_arguments('decay_rate_rho', args)
        mass_rho = _from_parameters_or_arguments('mass_rho', args)
        a_rho = _from_parameters_or_arguments('a_rho', args)
        decay_rate_phi_double_prime = _from_parameters_or_arguments('decay_rate_phi_double_prime', args)
        mass_phi_double_prime = _from_parameters_or_arguments('mass_phi_double_prime', args)
        decay_rate_phi_prime = _from_parameters_or_arguments('decay_rate_phi_prime', args)
        mass_phi_prime = _from_parameters_or_arguments('mass_phi_prime', args)
        a_phi_prime = _from_parameters_or_arguments('a_phi_prime', args)
        decay_rate_phi = _from_parameters_or_arguments('decay_rate_phi', args)
        mass_phi = _from_parameters_or_arguments('mass_phi', args)
        a_phi = _from_parameters_or_arguments('a_phi', args)
        decay_rate_omega_double_prime = _from_parameters_or_arguments('decay_rate_omega_double_prime', args)
        mass_omega_double_prime = _from_parameters_or_arguments('mass_omega_double_prime', args)
        a_omega_double_prime = _from_parameters_or_arguments('a_omega_double_prime', args)
        decay_rate_omega_prime = _from_parameters_or_arguments('decay_rate_omega_prime', args)
        mass_omega_prime = _from_parameters_or_arguments('mass_omega_prime', args)
        a_omega_prime = _from_parameters_or_arguments('a_omega_prime', args)
        decay_rate_omega = _from_parameters_or_arguments('decay_rate_omega', args)
        mass_omega = _from_parameters_or_arguments('mass_omega', args)
        a_omega = _from_parameters_or_arguments('a_omega', args)
        t_in_isovector = _from_parameters_or_arguments('t_in_isovector', args)
        t_in_isoscalar = _from_parameters_or_arguments('t_in_isoscalar', args)

        assert not args

        return function_form_factor(
            ts, parameters.t_0_isoscalar, parameters.t_0_isovector,
            t_in_isoscalar, t_in_isovector, a_omega, a_omega_prime, a_omega_double_prime,
            a_phi, a_phi_prime, a_rho, a_rho_prime, a_rho_double_prime, mass_omega, decay_rate_omega,
            mass_omega_prime, decay_rate_omega_prime, mass_omega_double_prime, decay_rate_omega_double_prime,
            mass_phi, decay_rate_phi, mass_phi_prime, decay_rate_phi_prime, mass_phi_double_prime,
            decay_rate_phi_double_prime, mass_rho, decay_rate_rho, mass_rho_prime,
            decay_rate_rho_prime, mass_rho_double_prime, decay_rate_rho_double_prime,
            mass_rho_triple_prime, decay_rate_rho_triple_prime)

    return partial_f


def _read_config(path_to_config):
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector
