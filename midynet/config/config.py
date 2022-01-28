from __future__ import annotations
import copy
import itertools
import pathlib
import pickle
import typing
import tqdm


from typing import Any, Callable
from collections import defaultdict
from .parameter import Parameter

__all__ = ["Config"]


class Config:

    separator: str = "."
    """ separator to distinguish """

    requirements: set[str] = {"name"}
    """ Set of required parameter names. """

    unique_parameters: set[str] = {"name"}
    """ Set of unique parameter names. """

    __cache__: bool = True
    """ Cache the generated sequences, makes the lookup faster at the expense of more memory. """

    def __init__(self, name="config", **kwargs):
        """
        Base config class containing and managing the parameters of an experiment set up.

        Operators:
            __str__, __repr__, __contains__, __getattr__, __getitem__, __len__,
            __gt__, __ge__, __lt__, __le__, __hash__

        Args:
            name: name of the configuration.
            **kwargs: other parameter contained by the configuration.
        """
        self.__parameters__ = {}
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None
        self.__hash_dict__ = None
        self.__subnames__ = {}
        self.__sequence__ = None
        self.__named_sequence__ = {}
        self.insert("name", name, unique=True)
        for k, v in kwargs.items():
            self.insert(
                k,
                v,
                unique=k in self.unique_parameters or self.unique_parameters == {"all"},
            )

    def __str__(self) -> str:
        s = self.__class__.__name__
        s += "("
        for k, v in self.items():
            if v.is_config and v.is_sequenced():
                s += f"{k}={[vv.name for vv in v.value]}, "
            elif v.is_config:
                s += f"{k}=`{v.value.name}`, "
            else:
                s += f"{k}=`{v.value}`, "
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
        return len(self.sequence())

    @classmethod
    def __auto__(cls, config_type, *others, **kwargs) -> Config:
        if config_type in cls.__dict__ and isinstance(
            getattr(cls, config_type), Callable
        ):
            return getattr(cls, config_type)(*others, **kwargs)
        elif isinstance(config_type, cls):
            return config_type
        else:
            message = f"Invalid config type `{type(config_type)}` for auto build of object `{cls.__name__}`."
            raise TypeError(message)

    def __gt__(self, other):
        return len(self) > len(other)

    def __ge__(self, other):
        return len(self) >= len(other)

    def __le__(self, other):
        return len(self) <= len(other)

    def __lt__(self, other):
        return len(self) < len(other)

    def __hash__(self) -> int:
        """
        Hash value of `self`.
        """
        if self.__self_hash__ is None:
            h = self.__compute_hash__()
            if self.__cache__:
                self.__self_hash__ = h
            else:
                return h
        return self.__self_hash__

    def __compute_hash__(self):
        if not self.is_sequenced():
            params = []
            for k, v in self.dict_copy().items():
                if v.is_config or v.is_unique():
                    continue
                params.append((k, hash(v)))
            return hash(tuple(params))
        else:
            message = "unhashable type, must not be sequenced."
            raise TypeError(message)

    @classmethod
    def auto(cls, config_type: str, *others, **kwargs) -> Config:
        """
        Automatic construction method using `args`.

        Args:
            args: input for constructing an instance of `Config`.
        """
        if not isinstance(config_type, typing.Iterable) or isinstance(config_type, str):
            return cls.__auto__(config_type, *others, **kwargs)
        else:
            return [cls.__auto__(c, *others, **kwargs) for c in config_type]

    def keys(self, recursively: bool = False) -> typing.KeysView:
        """
        Keys of the parameters.
        Args:
            recursively:    if `True`, recursvely returns the keys of all parameters
                            in `self`, else returns only the parameters directly
                            owned by `self`. Defaults to `False`.
        Returns:
            keys of the parameters.
        """
        if recursively:
            return self.dict_copy().keys()
        return self.__parameters__.keys()

    def values(self, recursively: bool = False) -> typing.ValuesView:
        """
        Values of the parameters.
        Args:
            recursively:    if `True`, recursvely returns the values of all parameters
                            in `self`, else returns only the parameters directly
                            owned by `self.` Defaults to `False`.
        Returns:
            values of the parameters.
        """
        if recursively:
            return self.dict_copy().values()
        return self.__parameters__.values()

    def items(self, recursively: bool = False) -> typing.ItemsView:
        """
        Items of the parameter dict.
        Args:
            recursively:    if `True`, recursvely returns the items of all parameters
                            in `self`, else returns only the parameters directly
                            owned by `self.` Defaults to `False`.
        Returns:
            items of the parameters.
        """
        if recursively:
            return self.dict_copy().items()
        return self.__parameters__.items()

    def unmet_requirements(self) -> set[str]:
        """
        Returns the set of missing requirements.
        """
        return self.requirements.difference(set(self.keys()))

    def is_sequenced(self) -> bool:
        """
        Check whether `self` contains sequenced parameters, otherwise returns False.
        """
        for v in self.values(True):
            if v.is_sequenced():
                return True
            elif v.is_config and v.value.is_sequenced():
                return True

        return False

    def is_equivalent(self, other: Config) -> bool:
        """
        Check whether `self` generated the same set of configurations with `other`, otherwise returns False.
        Args:
            other:  other configuration to compare with
        """
        if not issubclass(type(other), Config):
            return False

        s_self = self.format(forbid=self.unique_parameters).split("\n")[1:]
        s_other = other.format(forbid=self.unique_parameters).split("\n")[1:]
        return s_self == s_other

    def is_subset(self, other: Config) -> bool:
        """
        Check whether `self` generates a set of the configurations generated by `other`, otherwise returns False.

        Args:
            other:  other configuration to compare with
        """
        for c in self.sequence():
            if not c.is_subconfig(other):
                return False
        return True

    def is_subconfig(self, other: Config) -> bool:
        """
        Check whether `self`, which not be sequenced, is generated by `other`, otherwise returns False.

        Args:
            other:  other configuration to compare with
        """
        if self.is_sequenced():
            return False
        for k, v in other.hash_dict().items():
            if hash(self) in v:
                return True
        return False

    def insert(
        self,
        key: str,
        value: Any,
        **kwargs,
    ) -> None:
        """
        Insert a new parameter.

        Args:
            key:  key of the new parameter.
            value:  value of the new parameter.
            **kwargs: arguments past to :obj:`Parameter`.

        """
        value = value.value if issubclass(type(value), Parameter) else value
        p = Parameter(name=key, value=value, **kwargs)
        self.__parameters__[key] = p
        self.__parameters__[key].is_config = issubclass(p.datatype, Config)
        self.__reset_buffer__()

    def erase(self, key: str):
        """
        Erase existing parameters.

        Args:
            key: key of the parameter to erase.
        """
        self.__parameters__.pop(key)
        self.__reset_buffer__()

    def get_param(self, key: str, default: Any = None) -> Parameter:
        """
        Returns the parameter associated with a key.

        Args:
            key: key of the parameter to get.
            default (optional): Returned value if `key` is not in `self`. Defaults to None.
        """
        return self.dict_copy().get(key, default)

    def get_value(self, key: str, default: Any = None) -> typing.Any:
        """
        Returns the value associated with the parameter `key`.

        Args:
            key: key of the parameter to get.
            default (optional): Returned value if `key` is not in `self`. Defaults to None.
        """
        return self.get_param(key).value if key in self else default

    def set_value(self, key: str, value: Any) -> None:
        """
        Sets the value associated with the parameter `key`.

        Args:
            key: key of the parameter to set.
            value: new value of the parameter.
        """
        self.dict_copy().get(key).set_value(value)
        self.__reset_buffer__()

    # Copying methods

    def dict_copy(self, prefix="", recursively=True) -> typing.Dict[str, Parameter]:
        copy = {}
        """
        Generates a dictionary representation of the configuration, that looks up
        recursively the parameters values (if `recursively` is True).

        Args:
            prefix (optional): prefix of each key in the returned :obj:`dict`. Default is empty.
            recursvely: whether the dictionary include the parameters of all configs or not. Defaults to `True`.
        """

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

    def copy(self) -> Config:
        """
        Returns a shallow copy of the configuration.
        """
        config_copy = self.__class__()
        for k, v in self.dict_copy(recursively=False).items():
            config_copy.insert(
                k,
                v.value,
                unique=v.unique,
                with_repetition=v.with_repetition,
                force_non_sequence=v.force_non_sequence,
                sort_sequence=v.sort_sequence,
            )
        return config_copy

    def deepcopy(self) -> Config:
        """
        Returns a deep copy of the configuration.
        """
        config_copy = self.__class__()
        for k, v in self.dict_copy(recursively=False).items():
            config_copy.insert(
                k,
                copy.deepcopy(v.value),
                unique=v.unique,
                with_repetition=v.with_repetition,
                force_non_sequence=v.force_non_sequence,
                sort_sequence=v.sort_sequence,
            )
        return config_copy

    # Formating methods

    def save(
        self, path: typing.Union[str, pathlib.Path] = "config.pickle"
    ) -> pathlib.Path:
        """
        Saves the configuration in a pickle format.

        Args:
            path (optional): path where the :obj:`Config` is saved. Defaults to 'config.pickle'.
        """
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open(mode="wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: typing.Union[str, pathlib.Path] = "config.pickle") -> Config:
        """
        Loads a configuration from a pickle format and returns the associated configuration.

        Args:
            path (optional): path where to find the :obj:`Config` to load. Defaults to 'config.pickle'.
        """
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open(mode="rb") as f:
            c = pickle.load(f)
        return c

    def format(
        self,
        prefix: str = "",
        name_prefix: str = "",
        endline="\n",
        suffix="end",
        forbid: list = None,
    ) -> str:
        """
        Returns a string representation of the configuration, useful for debug.

        Args:
            prefix (optional): string put in front of each line.  Default is empty.
            name_prefix (optional): string put in front of each parameter.  Default is empty.
            endline (optional): string put at the end of each line. Default is standard end line.
            suffix (optional): string put at the end of the format. Default is 'end'.
            forbid (optional): parameters to forbid. Default is None.

        Notes:
            The prefix and name_prefix are used to format the :obj:`Config` recursively, and therefore
            should not be used in general.
        """
        s = f"{prefix}{self.__class__.__name__}(name=`{self.name}`): \n"

        forbid = [] if forbid is None else forbid
        for k, v in self.items():
            if k in forbid or k == "name":
                continue
            elif v.is_config and v.is_sequenced():
                s += f"{prefix}|\t{v.name}(["
                for i, c in enumerate(v.value):
                    s += f"name=`{c.name}`, "
                s = s[:-2] + f"]):{endline}"

                for i, c in enumerate(v.value):
                    ss = c.format(
                        prefix=prefix + f"|\t",
                        name_prefix=f"{c.name}{self.separator}",
                        endline=endline,
                        suffix=None,
                    )
                    ss = ss.split(":")[1:]
                    ss = "".join(ss)[2:]
                    ss = ss.split(endline)
                    s += endline.join(ss)
                s += f"{prefix}|\t{suffix}{endline}"
            elif v.is_config:
                format = v.value.format(prefix=prefix + "|\t")
                format = format.split(":")[1:]
                format = "".join(format)
                name = f"{v.name}(name=`{v.value.name}`)"
                ss = f"{prefix}|\t{name}:{format}{endline}".split("\n")
                if len(ss) > 3:
                    s += "\n".join(ss)
                else:
                    s += ss[0] + "\n"
            else:
                s += f"{prefix}|\t{name_prefix}{v.name} = {v.format()}{endline}"
        if suffix is not None and len(self.items()) > 0:
            s += f"{prefix}{suffix}"
        return s

    def merge_with(self, other: Config, verbose=0) -> None:
        """
        Merge config with other :obj:`Config`.

        Args:
            other: other :obj:`Config`.
        """

        other.__reset_buffer__()
        counter = 0
        size = len(other)
        if verbose:
            pbar = tqdm.tqdm(range(size), f"Merging {self.name} with {other.name}")
        for config in other.__generate_sequence__():
            counter += 1
            if config.is_subconfig(self):
                continue
            for key, value in config.items():
                if value.is_unique() or value.force_non_sequence:
                    continue
                elif key not in self:
                    message = f"Missing key {key}, configs cannot be merged."
                    raise ValueError(message)
                elif value.is_config:
                    if self.get_param(key).is_sequenced():
                        found = False
                        for sub in self.get_value(key):
                            if sub.name == value.value.name:
                                sub.merge_with(value.value, verbose=0)
                                found = True
                                break
                        if not found:
                            self.get_param(key).add_value(value.value)
                    else:
                        if value.value.name != self.get_value(key).name:
                            self.get_param(key).add_value(config.get_value(key))
                        else:
                            self.get_value(key).merge_with(
                                config.get_value(key), verbose=0
                            )
                else:
                    if self.get_param(key).is_sequenced() or value.is_sequenced():
                        self.get_param(key).add_values(value.value)
                    else:
                        self.get_param(key).add_value(value.value)
            if verbose:
                pbar.update()
        self.__reset_buffer__()

    # Methods that involve cache

    def __generate_sequence__(
        self, only: str = None
    ) -> typing.Generator[Config, None, None]:
        """
        Generates a sequence of the non-sequenced configurations, whose name is `only`,
        derived from `self`.

        Args:
            only: name of the :obj:`Configs` to generate.
        """
        keys_to_scan = []
        values_to_scan = []

        for k, v in self.items():
            if v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.generate_sequence())
            elif v.is_config and v.value.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.value.__generate_sequence__())
        if len(values_to_scan) == 0:
            yield self
        else:
            for values in itertools.product(*values_to_scan):
                config = self.copy()
                for k, v in zip(keys_to_scan, values):
                    config.set_value(k, v)
                    config.get_param(k).unique = self.get_param(k).unique
                    config.get_param(k).with_repetition = self.get_param(
                        k
                    ).with_repetition
                    config.get_param(k).force_non_sequence = self.get_param(
                        k
                    ).force_non_sequence
                    config.get_param(k).sort_sequence = self.get_param(k).sort_sequence
                if config.is_sequenced():
                    for c in config.__generate_sequence__():
                        name = self.subname(c)
                        if only is None or name == only:
                            c.set_value("name", name)
                            yield c
                else:
                    name = self.__compute_subname__(config)
                    if only is None or name == only:
                        config.set_value("name", name)
                        yield config

    def __reset_buffer__(self) -> None:
        """
        Resets the buffer of the variables contained by `self`. If `cache` is False, it does nothing.
        """
        self.__names__ = None
        self.__scanned_keys__ = None
        self.__scanned_values__ = None
        self.__sequence_size__ = None
        self.__hash_dict__ = None
        self.__self_hash__ = None
        self.__subnames__ = {}
        self.__sequence__ = None
        self.__named_sequence__ = {}

    def names(self) -> set[str]:
        """
        Set of all subnames generated by `self`.
        """
        if self.__names__ is None:
            names = set()
            for c in self.__generate_sequence__():
                names.add(c.name)
            if self.__cache__:
                self.__names__ = names
            else:
                return names
        return self.__names__

    def __compute_subname__(self, subconfig: Config) -> str:
        """
        Compute the subname of a subconfiguration.

        Args:
            subconfig: config from which we compute the subname.
        """
        ext = ""
        for k, v in subconfig.items():
            if not v.is_config:
                continue
            if self.get_param(k).is_sequenced():
                ext += f"{self.separator}{v.value.name}"
            elif self.get_value(k).is_sequenced():
                s = self.separator.join(
                    self.get_value(k)
                    .__compute_subname__(v.value)
                    .split(self.separator)[1:]
                )
                if len(s) > 0:
                    ext += self.separator + s
        return self.name + ext

    def scanned_keys(self) -> typing.Dict[str, list]:
        """
        Dictionary containing as keys all subnames and as values the keys associated
        with the sequenced parameters.
        """
        if self.__scanned_keys__ is None:
            counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for c in self.sequence():
                for k, v in c.items(recursively=True):
                    if not v.unique and not v.is_config:
                        counter[c.name][k][v.value] += 1
            keys = {k: [] for k in self.names()}
            for name, counter_dict in counter.items():
                for k, c in counter_dict.items():
                    if len(c) > 1:
                        keys[name].append(k)
            self.__scanned_keys__ = dict(keys)

        return self.__scanned_keys__

    def scanned_values(self) -> typing.Dict[str, typing.Dict[str, list]]:
        """
        Dictionary containing as keys all subnames and, as values, dictionaries
        that contain the values of the sequenced parameters.
        """
        if self.__scanned_values__ is None:
            values = {k: defaultdict(lambda: list()) for k in self.names()}
            keys = self.scanned_keys()
            if len(keys) == 0:
                self.__scanned_values__ = {}
            else:
                for c in self.sequence():
                    for k in keys[c.name]:
                        if c.get_value(k) not in values[c.name][k]:
                            values[c.name][k].append(c.get_value(k))
                values = dict({k: dict(v) for k, v in values.items()})
                if self.__cache__:
                    self.__scanned_values__ = values
                else:
                    return values
        return self.__scanned_values__

    def hash_dict(self) -> typing.Dict[str, list[int]]:
        """
        Dictionary containing as keys all subnames and, as value, the hash of all
        subconfigurations associated with the correspônding subname.
        """
        if self.__hash_dict__ is None:
            hash_dict = defaultdict(list)
            for name in self.names():
                for c in self.named_sequence(name):
                    hash_dict[name].append(hash(c))
            if self.__cache__:
                self.__hash_dict__ = dict(hash_dict)
            else:
                return dict(hash_dict)
        return self.__hash_dict__

    def sequence(self) -> list[Config]:
        """
        List of subconfigurations generated by `self`.
        """
        if self.__sequence__ is None:
            if self.__cache__:
                self.__sequence__ = [c for c in self.__generate_sequence__()]
            else:
                return [c for c in self.__generate_sequence__()]
        return self.__sequence__

    def named_sequence(self, subname) -> list[Config]:
        """
        List of subconfigurations generated by `self` with specific subname.
        """
        if subname not in self.__named_sequence__:
            if self.__cache__:
                self.__named_sequence__[subname] = [
                    c for c in self.__generate_sequence__(only=subname)
                ]
            else:
                return [c for c in self.__generate_sequence__(only=subname)]
        return self.__named_sequence__[subname]

    def subname(self, config) -> str:
        """
        Subname associated with a subconfiguration generated by `self`.
        """
        if hash(config) not in self.__subnames__:
            if self.__cache__:
                self.__subnames__[hash(config)] = self.__compute_subname__(config)
            else:
                return self.__compute_subname__(config)
        return self.__subnames__[hash(config)]


if __name__ == "__main__":
    pass
