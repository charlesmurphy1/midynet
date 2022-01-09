import itertools
import pathlib
import typing

from .parameter import Parameter
from typing import Any, Callable


class Config:
    separator: str = "/"
    requirements: set[str] = {"name"}

    def __init__(self, **kwargs):
        self.__parameters__ = {}
        for k, v in kwargs.items():
            self.insert(k, v)

    @classmethod
    def auto(cls, config):
        if isinstance(config, cls):
            return config
        else:
            message = f"Invalid type `{type(config)}` for auto build of object `{cls.__name__}`."
            raise TypeError(message)

    def keys(self, recursively: bool = False) -> typing.KeysView:
        if recursively:
            return self.dict_copy(recursively=True).keys()
        return self.__parameters__.keys()

    def values(self, recursively: bool = False) -> typing.ValuesView:
        if recursively:
            return self.dict_copy(recursively=True).values()
        return self.__parameters__.values()

    def items(self, recursively=False) -> typing.ItemsView:
        if recursively:
            return self.dict_copy(recursively=True).items()
        return self.__parameters__.items()

    def insert(self, key: str, value: Any):
        value = value.value if issubclass(type(value), Parameter) else value
        self.__parameters__[key] = Parameter(name=key, value=value)
        self.__parameters__[key].isConfig = issubclass(type(value), Config)

    def erase(self, key: str):
        self.__parameters__.pop(key)

    def is_sequenced(self) -> bool:
        for v in self.values():
            if v.is_sequenced():
                return True
        return False

    def is_equivalent(self, other) -> bool:
        for k, p in self.dict_copy(recursively=True).items():
            pp = other.get(k)
            if p.isConfig and not p.is_equivalent(pp):
                return False
            elif p.is_unique() and p.value != pp.value:
                return False
        return True

    def unmet_requirements(self):
        return self.requirements.difference(set(self.keys()))

    def is_subconfig(self, other) -> bool:
        for k, p in self.dict_copy(recursively=True).items():
            pp = other.get(k)
            if p.isConfig and not p.value.is_subconfig(pp.value):
                return False
            elif not p.isConfig and not p.is_unique() and p.is_sequenced():
                if not pp.is_sequenced() and pp.value not in p.value:
                    return False
                elif pp.is_sequenced() and not set(p.value).issubset(set(pp.value)):
                    return False
            elif not p.isConfig and not p.is_sequenced() and p.value != pp.value:
                return False
        return True

    def __str__(self) -> str:
        s = self.__class__.__name__
        s += "("
        for k, v in self.items():
            if v.isConfig:
                s += f"{k}={v.value.name}, "
            else:
                s += f"{k}={v.value}, "
        s = s[:-2] + ")" if s[-2:] == ", " else s + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key) -> bool:
        return key in self.keys()

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.__parameters__:
            return self.__parameters__[key].value
        else:
            message = f"This config has no attribute `{key}`"
            raise AttributeError(message)

    def __setattr__(self, key, value):
        if "__parameters__" in self.__dict__ and key in self.__parameters__:
            self.__parameters__[key].set(value)
        else:
            self.__dict__[key] = value

    def __setitem__(self, key, value):

        if not isinstance(value, self.__parameters__[key].datatype):
            message = f"type `{type(value)}` of key `{key}` is invalid, expected type `{self.__parameters__[key].datatype}`."
            raise TypeError(message)

        if key in self:
            self.__parameters__[key].value = value
        else:
            self.insert(key, value)

    def __getitem__(self, key: str) -> Parameter:
        if key not in self:
            message = f"key `{key}` has not been found."
            raise LookupError(message)
        return self.__parameters__[key]

    def get(self, key: str) -> Parameter:
        path = key.split(self.separator)
        key = path[0]
        if self.__parameters__[key].isConfig and len(path) > 1:
            return self.__parameters__[key].value.get(self.separator.join(path[1:]))
        else:
            return self.__parameters__[key]

    def set(self, key: str, value: Any):
        path = key.split(self.separator)
        key = path[0]
        if self.__parameters__[key].isConfig and len(path) > 1:
            self.__parameters__[key].value.set(self.separator.join(path[1:]), value)
        else:
            self.__parameters__[key].set(value)

    def dict_copy(self, recursively=False, prefix="") -> typing.Dict[str, Parameter]:
        copy = {}

        for k, v in self.items():
            copy[prefix + k] = v
            if v.isConfig and recursively:
                copy.update(
                    v.value.dict_copy(
                        recursively=recursively,
                        prefix=prefix + k + self.separator,
                    )
                )
        return copy

    def copy(self):
        return Config(**self.dict_copy())

    def format(self, prefix: str = "") -> str:
        s = prefix + self.name + ": \n"
        for k, v in self.items():
            if v.isConfig:
                v_format = v.value.format(prefix=prefix + "|\t ")
                v_format = v_format.split(":")[1:]
                v_string = "".join(v_format)
                s += f"{prefix}|\t {v.name}({v.value.name}):{v_string}\n"
            else:
                s += f"{prefix}|\t {v.name}={v.value}\n"
        s += f"{prefix}end"
        return s

    def generate_sequence(self) -> typing.Generator:
        keys_to_scan = []
        values_to_scan = []
        for k, v in self.items():
            if v.isConfig and v.value.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.value.generate_sequence())
            elif v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.generate_sequence())
        for values in itertools.product(*values_to_scan):
            config = self.copy()
            for k, v in zip(keys_to_scan, values):
                config.set(k, v)
            yield config

    def merge(self, other):
        raise NotImplementedError()

    def save(self, path: pathlib.Path):
        raise NotImplementedError()

    @staticmethod
    def load(path: pathlib.Path):
        raise NotImplementedError()


if __name__ == "__main__":
    pass


# class ConfigArray:
#
#     def merge(self, config):
#         assert issubclass(
#             config.__class__, (Config, ConfigArray)
#         ), f"Invalid object {config.__class__} for other."
#         new_config = self.config.copy()
#         for k, v in self.state_dict.items():
#             if k not in config.state_dict:
#                 continue
#             vv = config.state_dict[k]
#             if isinstance(v, (list, tuple)):
#                 if not isinstance(vv, (list, tuple)):
#                     vv = [vv]
#                 new_config[k] = tuple(dict.fromkeys(list(v) + list(vv)))
#             elif v != vv:
#                 new_config[k] = (v, vv)
#             if isinstance(new_config[k], tuple):
#                 if k in self.labels_to_scan:
#                     new_config[k] = list(sorted(new_config[k]))
#                 else:
#                     new_config[k] = tuple(sorted(new_config[k]))
#         self.config = new_config
#
#     def batch(self, size):
#         l = len(self)
#         for i in range(0, l, size):
#             yield self[i : min(i + size, l)]
