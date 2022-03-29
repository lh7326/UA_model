from configparser import ConfigParser
from kaon_production.function import function_form_factor
from kaon_production.ModelParameters import ModelParameters


def make_partial_for_parameters(parameters: ModelParameters):
    def _build_parameters_scheme():
        argument_index = 0
        scheme = []
        for parameter in parameters:
            if parameter.is_fixed:
                scheme.append(lambda _: parameter.value)
            else:
                scheme.append(lambda args: args[argument_index])
                argument_index += 1
        return scheme, argument_index

    scheme, args_length = _build_parameters_scheme()

    def partial_f(ts, *args):
        assert len(args) == args_length
        evaluated_scheme = [val(args) for val in scheme]
        return function_form_factor(
            ts, parameters.t_0_isoscalar, parameters.t_0_isovector, *evaluated_scheme
        )

    return partial_f


def _read_config(path_to_config):
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(path_to_config)

    pion_mass = config.getfloat('constants', 'charged_pion_mass')
    t_0_isoscalar = (3 * pion_mass) ** 2
    t_0_isovector = (2 * pion_mass) ** 2
    return t_0_isoscalar, t_0_isovector
