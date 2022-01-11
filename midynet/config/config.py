import itertools
import pathlib
import pickle
import typing

from typing import Any, Callable
from collections import defaultdict
from .parameter import Parameter


class Config:
    """
    Base config class containing the parameters of an experiment set up.
    """

    separator: str = "."
    requirements: set[str] = {"name"}
    unique_parameters: set[str] = {"name"}

    def __init__(self, **kwargs):
        self.__parameters__ = {}
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None
        if "name" not in kwargs:
            kwargs["name"] = "config"
        for k, v in kwargs.items():
            self.insert(k, v, unique=k in self.unique_parameters)

    @classmethod
    def __auto__(cls, args):
        """ Automatic construction method. """
        if args in cls.__dict__ and isinstance(getattr(cls, args), Callable):
            return getattr(cls, args)()
        elif isinstance(args, cls):
            return args
        else:
            message = f"Invalid type `{type(args)}` for auto build of object `{cls.__name__}`."
            raise TypeError(message)

    @classmethod
    def auto(cls, args):
        if not isinstance(args, typing.Iterable) or isinstance(args, str):
            return cls.__auto__(args)
        else:
            return [cls.__auto__(a) for a in args]

    def __str__(self) -> str:
        s = self.__class__.__name__
        s += "("
        for k, v in self.items():
            if v.is_config:
                s += f"{k} = `{v.value.name}`, "
            else:
                s += f"{k} = `{v.value}`, "
        s = s[:-2] + ")" if s[-2:] == ", " else s + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key) -> bool:
        return key in self.keys()

    def __getattr__(self, key):
        if key == "__parameters__" and key not in self.__dict__:
            self.__dict__[key] = {}

        if key in getattr(self, "__parameters__"):
            return self.get_value(key)
        elif key in self.__dict__:
            return getattr(self, key)
        else:
            message = f"This config has no attribute `{key}`"
            raise AttributeError(message)

    def __getitem__(self, key: str) -> Parameter:
        if key not in self:
            message = f"key `{key}` has not been found."
            raise LookupError(message)
        return self.get_param(key)

    def keys(self, recursively: bool = False) -> typing.KeysView:
        """ Keys of the parameters. """
        if recursively:
            return self.dict_copy(recursively=True).keys()
        return self.__parameters__.keys()

    def values(self, recursively: bool = False) -> typing.ValuesView:
        """ Values of the parameters. """
        if recursively:
            return self.dict_copy(recursively=True).values()
        return self.__parameters__.values()

    def items(self, recursively=False) -> typing.ItemsView:
        """ Items of the parameter dict. """
        if recursively:
            return self.dict_copy(recursively=True).items()
        return self.__parameters__.items()

    def reset_buffer(self):
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None

    def insert(
        self,
        key: str,
        value: Any,
        unique: bool = False,
        with_repetition: bool = False,
        force_non_sequence: bool = False,
    ):
        """ Insert new parameters. """
        value = value.value if issubclass(type(value), Parameter) else value
        p = Parameter(
            name=key, value=value, unique=unique, force_non_sequence=force_non_sequence
        )
        self.__parameters__[key] = p
        self.__parameters__[key].is_config = issubclass(p.datatype, Config)
        self.__parameters__[key].with_repetition = (
            self.__parameters__[key].is_config or with_repetition
        )
        self.reset_buffer()

    def erase(self, key: str):
        """ Erase existing parameters. """
        self.__parameters__.pop(key)
        self.reset_buffer()

    def is_sequenced(self) -> bool:
        for v in self.values():
            if v.is_sequenced():
                return True
            elif v.is_config:
                return v.value.is_sequenced()
        return False

    def is_equivalent(self, other) -> bool:
        if not issubclass(type(other), Config):
            return False

        return self.format() == other.format()

    def unmet_requirements(self):
        return self.requirements.difference(set(self.keys()))

    def is_subconfig(self, other) -> bool:
        if not issubclass(type(other), Config):
            return False
        for k, p in self.dict_copy(recursively=True).items():
            pp = other.get_param(k)
            if p.is_config and not p.value.is_subconfig(pp.value):
                return False
            elif not p.is_config and not p.is_unique() and p.is_sequenced():
                if not pp.is_sequenced() and pp.value not in p.value:
                    return False
                elif pp.is_sequenced() and not set(p.value).issubset(set(pp.value)):
                    return False
            elif not p.is_config and not p.is_sequenced() and p.value != pp.value:
                return False
        return True

    def get_param(self, key: str, default: Any = None) -> Parameter:
        path = key.split(self.separator)
        key = path[0]
        if key not in self.__parameters__:
            if default is None:
                default = Config()
            return default
        elif self.__parameters__[key].is_config and len(path) > 1:
            return self.__parameters__[key].value.get_param(
                self.separator.join(path[1:]), default=default
            )
        else:
            return self.__parameters__[key]

    def get_value(self, key: str, default: Any = None):
        return self.get_param(key).value

    def set_value(self, key: str, value: Any):
        path = key.split(self.separator)
        key = path[0]
        if self.__parameters__[key].is_config and len(path) > 1:
            self.__parameters__[key].value.set_value(
                self.separator.join(path[1:]), value
            )
        else:
            self.__parameters__[key].set_value(value)
        self.reset_buffer()

    def dict_copy(self, recursively=False, prefix="") -> typing.Dict[str, Parameter]:
        copy = {}

        for k, v in self.items():
            copy[prefix + k] = v
            if v.is_config and recursively:
                copy.update(
                    v.value.dict_copy(
                        recursively=recursively,
                        prefix=prefix + k + self.separator,
                    )
                )
        return copy

    def copy(self):
        return Config(**self.dict_copy())

    def format(
        self, prefix: str = "", endline="\n", suffix="end\n", forbid: list = None
    ) -> str:
        s = f"{prefix + self.__class__.__name__}(name=`{self.name}`): \n"

        forbid = ["name"] if forbid is None else forbid
        for k, v in self.items():
            if k in forbid:
                continue
            elif v.is_config and v.is_sequenced():
                s += f"{prefix}|\t {v.name}(["
                for i, c in enumerate(v.value):
                    s += f"name=`{c.name}`, "
                s = s[:-2] + f"]):{endline}"

                for i, c in enumerate(v.value):
                    ss = c.format(
                        prefix=prefix + f"|\t",
                        endline=endline,
                        suffix=None,
                    )
                    ss = ss.split(":")[1:]
                    ss = "".join(ss)[2:]
                    ss = ss.split(endline)[:-1]
                    for i, _ in enumerate(ss):
                        ss[i] += f"\t (name=`{c.name}`)"
                    s += endline.join(ss) + endline
            elif v.is_config:
                format = v.value.format(prefix=prefix + "|\t ")
                format = format.split(":")[1:]
                format = "".join(format)[:-1]
                name = f"{v.name}(name=`{v.value.name}`)"
                s += f"{prefix}|\t {name}:{format}{endline}"
            elif v.is_config and v.is_sequenced():
                s += format_subconfig(v, v.value)

            else:
                s += f"{prefix}|\t {v.name} = `{v.value}`{endline}"
        if suffix is not None:
            s += f"{prefix}{suffix}"
        return s

    def generate_sequence(self) -> typing.Generator:
        keys_to_scan = []
        values_to_scan = []

        for k, v in self.items():
            if v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.generate_sequence())
            elif v.is_config and v.value.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.value.generate_sequence())

        if len(values_to_scan) == 0:
            yield self

        for values in itertools.product(*values_to_scan):
            config = self.copy()

            name = config.name
            for k, v in zip(keys_to_scan, values):
                config.set_value(k, v)
                if issubclass(type(v), Config):
                    name += self.separator + v.name
            if config.is_sequenced():
                for c in config.generate_sequence():
                    c.set_value("name", name)
                    yield c
            else:
                config.set_value("name", name)
                yield config
        return

    @property
    def names(self):
        if self.__names__ is None:
            self.__names__ = set()
            for c in self.generate_sequence():
                self.__names__.add(c.name)
        return self.__names__

    @property
    def scanned_keys(self) -> typing.Dict[str, list]:
        if self.__scanned_keys__ is None:
            counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for c in self.generate_sequence():
                for k, v in c.items(recursively=True):
                    if not v.unique and not v.is_config:
                        counter[c.name][k][v.value] += 1
            keys = defaultdict(lambda: list())
            for name, counter_dict in counter.items():
                for k, c in counter_dict.items():
                    if len(c) > 1:
                        keys[name].append(k)
            self.__scanned_keys__ = dict(keys)
        return self.__scanned_keys__

    @property
    def scanned_values(self) -> typing.Dict[str, typing.Dict[str, list]]:
        if self.__scanned_values__ is None:
            values = defaultdict(lambda: defaultdict(lambda: list()))
            keys = self.scanned_keys
            for c in self.generate_sequence():
                for k in keys[c.name]:
                    if c.get_value(k) not in values[c.name][k]:
                        values[c.name][k].append(c.get_value(k))
            self.__scanned_values__ = dict({k: dict(v) for k, v in values.items()})
        return self.__scanned_values__

    def merge(self, other):
        for k, v in self.items():
            if not v.unique:
                if v.is_config:
                    v.merge(other.get_value(k))
                else:
                    v.add_values(other.get_value(k))

    def save(
        self, path: typing.Union[str, pathlib.Path] = "config.pickle"
    ) -> pathlib.Path:
        path = pathlib.Path(path) if path is str else path
        with path.open(mode="wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: typing.Union[str, pathlib.Path] = "config.pickle"):
        path = pathlib.Path(path) if path is str else path
        with path.open(mode="rb") as f:
            config = pickle.load(f)
        return config


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
