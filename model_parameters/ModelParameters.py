from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Iterator, List, Tuple


Parameter = namedtuple('Parameter', 'name value is_fixed')


class ModelParameters(ABC):

    def __init__(self, *args, always_fixed: Tuple[str, ...] = (), **kwargs) -> None:
        self._always_fixed = always_fixed
        self._data = self._setup_data(*args, **kwargs)

    @abstractmethod
    def _setup_data(self, *args, **kwargs) -> List[Parameter]:
        """
        Prepare the list of parameters in which we internally store the parameter data.
        The members of the list are of the type Parameter (a named tuple defined above)
        and their ordering is significant.

        This method is called in the initialization procedure.

        """
        pass

    @classmethod
    @abstractmethod
    def from_list(cls, list_of_parameters: List[Parameter]) -> 'ModelParameters':
        pass

    def to_list(self) -> List[Parameter]:
        return list(self._data)

    def _find(self, name: str) -> Tuple[int, Parameter]:
        for index, parameter in enumerate(self._data):
            if parameter.name == name:
                return index, parameter
        raise KeyError(f'No such key: {name}')

    def __getitem__(self, item: str) -> Parameter:
        _, parameter = self._find(item)
        return parameter

    def __setitem__(self, key: str, new_value: Parameter) -> None:
        index, parameter = self._find(key)
        if not isinstance(new_value, Parameter):
            raise TypeError('Bad type: new_value must be of type Parameter!')
        if not parameter.name == new_value.name:
            raise ValueError('When setting new parameters names must be preserved!')
        if new_value.name in self._always_fixed and not new_value.is_fixed:
            raise ValueError(f'The parameter {new_value.name} must always be fixed!')
        self._data[index] = new_value

    def set_value(self, parameter_name: str, new_parameter_value: float) -> None:
        index, old_par = self._find(parameter_name)
        new_par = Parameter(name=parameter_name, value=new_parameter_value, is_fixed=old_par.is_fixed)
        self._data[index] = new_par

    def fix_parameters(self, names: List[str]) -> None:
        for name in names:
            index, parameter = self._find(name)
            self._data[index] = Parameter(parameter.name, parameter.value, True)

    def release_parameters(self, names: List[str]) -> None:
        for name in names:
            if name in self._always_fixed:
                raise ValueError(f'The parameter {name} cannot be released!')
        for name in names:
            index, parameter = self._find(name)
            self._data[index] = Parameter(parameter.name, parameter.value, False)

    def fix_all_parameters(self) -> None:
        for index, parameter in enumerate(self._data):
            self._data[index] = Parameter(parameter.name, parameter.value, True)

    def release_all_parameters(self) -> None:
        for index, parameter in enumerate(self._data):
            if parameter.name not in self._always_fixed:
                self._data[index] = Parameter(parameter.name, parameter.value, False)

    def get_fixed_values(self) -> List[float]:
        return [parameter.value for parameter in self._data if parameter.is_fixed]

    def get_free_values(self) -> List[float]:
        return [parameter.value for parameter in self._data if not parameter.is_fixed]

    def get_all_values(self) -> List[float]:
        return list(map(lambda p: p.value, self._data))

    def update_free_values(self, new_values: List[float]) -> None:
        free_parameters = [(i, parameter) for i, parameter
                           in enumerate(self._data) if not parameter.is_fixed]
        if len(free_parameters) != len(new_values):
            raise ValueError(f'Wrong number of new values: got {len(new_values)}, requires {len(free_parameters)}.')
        for (i, parameter), new_value in zip(free_parameters, new_values):
            self._data[i] = Parameter(parameter.name, new_value, parameter.is_fixed)

    def __iter__(self) -> Iterator[Parameter]:
        for parameter in self._data:
            yield parameter

    @abstractmethod
    def get_bounds_for_free_parameters(self, *args, **kwargs) -> Tuple[List[float], List[float]]:
        pass
