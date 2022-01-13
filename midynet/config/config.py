import copy
import itertools
import pathlib
import pickle
import typing

from typing import Any, Callable
from collections import defaultdict
from .parameter import Parameter

__all__ = ["Config"]


class Config:
    """
    Base config class containing the parameters of an experiment set up.
    """

    separator: str = "."
    requirements: set[str] = {"name"}
    unique_parameters: set[str] = {"name"}

    def __init__(self, name="config", **kwargs):
        self.__parameters__ = {}
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None
        self.__sequence_size__ = None
        self.__dict_copy__ = None
        self.insert("name", name, unique=True)
        for k, v in kwargs.items():
            self.insert(k, v, unique=k in self.unique_parameters)

    def __str__(self) -> str:
        s = self.__class__.__name__
        s += "("
        for k, v in self.items():
            if v.is_config and v.is_sequenced():
                s += f"{k}={[vv.name for vv in v.value]}, "
            elif v.is_config:
                s += f"{k}={v.value.name}, "
            else:
                s += f"{k}={v.value}, "
        s = s[:-2] + ")" if s[-2:] == ", " else s + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key) -> bool:
        return key in self.keys(recursively=True)

    def __getattr__(self, key):
        if key == "__parameters__" and key not in self.__dict__:
            self.__dict__[key] = {}

        if key in getattr(self, "__parameters__"):
            return self.get_value(key)
        elif key in self.__dict__:
            return getattr(self, key)
        else:
            message = f"Config `{self}` has no attribute `{key}`"
            raise AttributeError(message)

    def __getitem__(self, key: str) -> Parameter:
        if key not in self:
            message = f"key `{key}` has not been found."
            raise LookupError(message)
        return self.get_param(key)

    def __len__(self):
        return self.sequence_size

    @classmethod
    def __auto__(cls, args):
        if args in cls.__dict__ and isinstance(getattr(cls, args), Callable):
            return getattr(cls, args)()
        elif isinstance(args, cls):
            return args
        else:
            message = f"Invalid type `{type(args)}` for auto build of object `{cls.__name__}`."
            raise TypeError(message)

    def __gt__(self, other):
        return len(self) > len(other)

    def __ge__(self, other):
        return len(self) >= len(other)

    def __le__(self, other):
        return len(self) <= len(other)

    def __lt__(self, other):
        return len(self) < len(other)

    def __hash__(self):
        if not self.is_sequenced():
            params = []
            for k, v in self.dict_copy().items():
                if v.is_config or v.is_unique():
                    continue
                params.append((k, v.value))
            return hash(tuple(params))
        else:
            message = "unhashable type, must not be sequenced."
            raise TypeError(message)

    @classmethod
    def auto(cls, args):
        """ Automatic construction method. """
        if not isinstance(args, typing.Iterable) or isinstance(args, str):
            return cls.__auto__(args)
        else:
            return [cls.__auto__(a) for a in args]

    def keys(self, recursively: bool = False) -> typing.KeysView:
        """ Keys of the parameters. """
        if recursively:
            return self.dict_copy().keys()
        return self.__parameters__.keys()

    def values(self, recursively: bool = False) -> typing.ValuesView:
        """ Values of the parameters. """
        if recursively:
            return self.dict_copy().values()
            return self.dict_copy().values()
        return self.__parameters__.values()

    def items(self, recursively=False) -> typing.ItemsView:
        """ Items of the parameter dict. """
        if recursively:
            return self.dict_copy().items()
        return self.__parameters__.items()

    def reset_buffer(self):
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None
        self.__sequence_size__ = None
        self.__hashing_keys__ = None

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
            elif v.is_config and v.value.is_sequenced():
                return True
        return False

    def is_equivalent(self, other) -> bool:
        if not issubclass(type(other), Config):
            return False

        s_self = self.format(forbid=self.unique_parameters).split("\n")[1:]
        s_other = other.format(forbid=self.unique_parameters).split("\n")[1:]
        return s_self == s_other

    def unmet_requirements(self):
        return self.requirements.difference(set(self.keys()))

    def is_subset(self, other) -> bool:
        for c in other.generate_sequence():
            if not self.is_subconfig(c):
                return False
        return True

    def is_subconfig(self, other) -> bool:
        if other.is_sequenced():
            return False
        return hash(other) in self.hashing_keys

    def get_param(self, key: str, default: Any = None) -> Parameter:
        return self.dict_copy().get(key, default)

    def get_value(self, key: str, default: Any = None):
        return self.get_param(key).value if key in self else default

    def set_value(self, key: str, value: Any):
        self.dict_copy().get(key).set_value(value)
        self.reset_buffer()

    def dict_copy(self, prefix="", recursively=True) -> typing.Dict[str, Parameter]:
        copy = {}

        for k, v in self.items():
            copy[f"{prefix}{k}"] = v
            if v.is_config and recursively:
                if v.is_sequenced():
                    for vv in v.value:
                        copy[f"{prefix}{k}{self.separator}{vv.name}"] = Parameter(
                            name=v.name, value=vv, unique=v.unique
                        )
                        copy.update(
                            vv.dict_copy(
                                prefix=f"{prefix}{k}{self.separator}{vv.name}{self.separator}",
                                recursively=recursively,
                            )
                        )
                else:
                    copy.update(
                        v.value.dict_copy(prefix=f"{prefix}{k}{self.separator}")
                    )
        return copy

    def copy(self):
        return self.__class__(**self.dict_copy(recursively=False))

    def deepcopy(self):
        params = {
            k: copy.deepcopy(v) for k, v in self.dict_copy(recursively=False).items()
        }
        return self.__class__(**params)

    def format(
        self,
        prefix: str = "",
        name_prefix: str = "",
        endline="\n",
        suffix="end",
        forbid: list = None,
    ) -> str:
        s = f"{prefix + self.__class__.__name__}(name={self.name}): \n"

        forbid = [] if forbid is None else forbid
        for k, v in self.items():
            if k in forbid:
                continue
            elif v.is_config and v.is_sequenced():
                s += f"{prefix}|\t {v.name}(["
                for i, c in enumerate(v.value):
                    s += f"name={c.name}, "
                s = s[:-2] + f"]):{endline}"

                for i, c in enumerate(v.value):
                    ss = c.format(
                        prefix=prefix + f"|\t",
                        name_prefix=c.name + self.separator,
                        endline=endline,
                        suffix=None,
                    )
                    ss = ss.split(":")[1:]
                    ss = "".join(ss)[2:]
                    ss = ss.split(endline)[:-1]
                    s += endline.join(ss) + endline
            elif v.is_config:
                format = v.value.format(prefix=prefix + "|\t ")
                format = format.split(":")[1:]
                format = "".join(format)[:-1]
                name = f"{v.name}(name={v.value.name})"
                s += f"{prefix}|\t {name}:{format}{endline}"
            else:
                s += f"{prefix}|\t {name_prefix}{v.name} = {v.value}{endline}"
        if suffix is not None:
            s += f"{prefix}{suffix}"
        return s

    def generate_sequence(self, only: str = None) -> typing.Generator:
        keys_to_scan = []
        _keys_to_scan = []
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
        else:
            for values in itertools.product(*values_to_scan):
                config = self.copy()
                name = config.name
                for k, v in zip(keys_to_scan, values):
                    config.set_value(k, v)
                    if issubclass(type(v), Config) and self.get_param(k).is_sequenced():
                        name += self.separator + v.name
                if config.is_sequenced():
                    for c in config.generate_sequence():
                        if only is None or name == only:
                            c.set_value("name", name)
                            yield c
                else:
                    config.set_value("name", name)
                    if only is None or name == only:
                        yield config

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
            if len(keys) == 0:
                self.__scanned_values__ = {}
            else:
                for c in self.generate_sequence():
                    for k in keys[c.name]:
                        if c.get_value(k) not in values[c.name][k]:
                            values[c.name][k].append(c.get_value(k))
                self.__scanned_values__ = dict({k: dict(v) for k, v in values.items()})
        return self.__scanned_values__

    @property
    def sequence_size(self) -> int:
        if self.__sequence_size__ is None:
            self.__sequence_size__ = sum(1 for _ in self.generate_sequence())
        return self.__sequence_size__

    @property
    def hashing_keys(self):
        if self.__hashing_keys__ is None:
            self.__hashing_keys__ = []
            for c in self.generate_sequence():
                self.__hashing_keys__.append(hash(c))
        return self.__hashing_keys__

    def regroup_by_name(self, name=None):
        if name is None:
            name = self.names
        elif isinstance(name, str):
            name = [name]

        config = defaultdict(list)
        for c in self.generate_sequence():
            config[c.name].append(c)
        return dict(config)

    def merge(self, other):
        for config in other.generate_sequence():
            if self.is_subconfig(config):
                continue
            for key, value in config.items():
                if value.is_unique():
                    continue
                elif key not in self:
                    message = f"Missing key {key}, configs cannot be merged."
                    raise ValueError(message)
                elif value.is_config and self.get_param(key).is_sequenced():
                    found = False
                    for sub in self.get_value(key):
                        if sub.name == value.value.name:
                            sub.merge(value.value)
                            found = True
                            break
                    if not found:
                        self.get_param(key).add_value(value.value)
                elif value.is_config:
                    if value.value.name != self.get_value(key).name:
                        self.get_param(key).add_value(config.get_value(key))
                    else:
                        value.value.merge(config.get_value(key))
                else:
                    self.get_param(key).add_values(value.value)
        self.reset_buffer()

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
