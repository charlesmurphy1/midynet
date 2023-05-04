from __future__ import annotations

from itertools import product
from typing import Any, List, Type, Optional
from pyhectiqlab import Config as BaseConfig
from copy import deepcopy
import functools
import pickle


def static(cls):
    @functools.wraps(cls, updated=())
    class StaticConfig(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            self.lock_types()

    return StaticConfig


def frozen(cls):
    @functools.wraps(cls, updated=())
    class FrozenConfig(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            self.lock()

    return FrozenConfig


class Config(BaseConfig):
    separator: str = "."

    def __init__(self, name: str, as_seq: bool = True, **kwargs: Any):
        self._name = name
        self._type_lock: bool = False
        self.__types__: dict[str, Type] = {}
        self.__sequence_params__: list[str] = []
        self._as_seq = as_seq
        super().__init__(**kwargs)
        self.name: str = name

    def __setattr__(self, name: str, value: Any) -> None:
        if name in [
            "_name",
            "_state",
            "_lock",
            "_type_lock",
            "_as_seq",
            "__types__",
            "__sequence_params__",
        ]:
            object.__setattr__(self, name, value)
        else:
            if self._lock:
                raise Exception("Config is locked.")
            if self._type_lock and name not in self._state:
                raise Exception(
                    f"In type-lock mode, new param `{name}` cannot be added."
                )
            if self._type_lock and not self.is_type(value, self.type(name)):
                raise Exception(
                    f"In type-lock mode, `{name}` must be of type `{self.type(name).__name__}`."
                )
            if isinstance(value, list) and not self.is_one_type(value):
                raise ValueError(
                    f"Value is `list` but contains multiple types: `{[type(v).__name__ for v in value]}`."
                )
            self._state[name] = value
            self.__types__[name] = (
                type(value) if not isinstance(value, list) else type(value[0])
            )
            if isinstance(value, list):
                self.__types__[name] = type(value[0])
                if self._as_seq:
                    self.__sequence_params__.append(name)
            else:
                self.__types__[name] = type(value)
                if name in self.__sequence_params__:
                    self.__sequence_params__.remove(name)

    def type(self, name: str) -> Type:
        return self.__types__[name]

    @property
    def name(self):
        return self._name

    @staticmethod
    def is_type(value: Any, expected_type: Type) -> bool:
        return isinstance(value, expected_type) or (
            isinstance(value, list)
            and all([isinstance(v, expected_type) for v in value])
        )

    @staticmethod
    def is_one_type(value: bool) -> bool:
        if not issubclass(value.__class__, list) or len(value) == 0:
            return True
        t = type(value[0])
        return all([isinstance(v, t) for v in value])

    def lock(self) -> None:
        self._lock = self._type_lock = True

    def unlock(self) -> None:
        self._lock = self._type_lock = False

    def lock_types(self) -> None:
        self._type_lock = True

    def unlock_types(self) -> None:
        self._type_lock = False

    def update(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self._state[key]
        components = key.split(self.separator)
        if components[0] in self:
            subkey = self.separator.join(components[1:])
            assert not isinstance(
                self._state[components[0]], (list, tuple, set)
            ), f"In get: Component {components[0]} of key {key} must not be iterable."
            return self._state[components[0]].get(subkey, default)

        return default

    def __len__(self) -> int:
        return len(list(self.to_sequence()))

    @property
    def dict(self) -> dict[str, Any]:
        """A property that converts a config to a dict. Supports nested Config."""
        d = {}
        for key, item in self._state.items():
            if issubclass(item.__class__, Config):
                d[key] = item.dict
            elif issubclass(item.__class__, list):
                d[key] = [
                    i.dict if issubclass(i.__class__, Config) else i for i in item
                ]
            else:
                d[key] = item
        return d

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        as_seq: bool = True,
    ) -> Config:
        """Convert a dict to a Config object.
        `data` can be nested (dict within dict), which will generate sub configs.
        If `data` is not a dict, then the method returns `data`.

        Example:
        -------------
        d = {"a": {"b": 4}}
        config = Config.from_dict(d)
        assert config.a.b == 4
        """
        if isinstance(data, dict) == False:
            return data

        config = Config(data.get("name", "generic"), as_seq=as_seq)
        for key in data:
            if isinstance(data[key], dict):
                config[key] = Config.from_dict(data[key])
            elif issubclass(data[key].__class__, list) and isinstance(
                data[key][0], dict
            ):
                config[key] = [Config.from_dict(d) for d in data[key]]
            else:
                config[key] = data[key]
            if isinstance(config[key], list):
                config.as_sequence(key)
        config.lock()
        config.lock_types()
        return config

    @classmethod
    def auto(
        cls,
        config: Optional[str or Config or List[Config]],
        *args: Any,
        **kwargs: Any,
    ) -> Config or list[Config]:
        if config is None:
            return
        configs = [config] if not isinstance(config, list) else config
        res = []
        for config in configs:
            if config in dir(cls):
                res.append(getattr(cls, config)(*args, **kwargs))
            elif isinstance(config, cls):
                res.append(config)
            else:
                t = config if isinstance(config, str) else type(config)
                message = f"Invalid config type `{t}` for auto build of object `{cls.__name__}`."
                raise TypeError(message)
        if len(res) == 1:
            return res[0]
        elif len(res) == 0:
            return
        return res

    def is_sequenced(self) -> bool:
        for k, v in self._state.items():
            if self.is_sequence(k):
                return True
            elif isinstance(v, Config) and v.is_sequenced():
                return True
        return False

    def as_sequence(self, key: str) -> None:
        if self.is_sequence(key):
            return
        if isinstance(self._state[key], (list, tuple, set)):
            self.__sequence_params__.append(key)

    def not_sequence(self, key: str) -> None:
        if not self.is_sequence(key):
            return
        if key in self.__sequence_params__:
            self.__sequence_params__.remove(key)

    def is_sequence(self, key: str) -> bool:
        return key in self.__sequence_params__

    def to_sequence(self):
        if not self.is_sequenced():
            yield self
            return

        keys_to_scan = []
        values_to_scan = []

        for k, v in self._state.items():
            if self.is_sequence(k):
                keys_to_scan.append(k)
                values_to_scan.append(v)
            elif issubclass(v.__class__, Config) and v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.to_sequence())
        for values in product(*values_to_scan):
            config = self.copy()
            config.unlock()
            for k, v in zip(keys_to_scan, values):
                setattr(config, k, v)
            if config.is_sequenced():
                for c in config.to_sequence():
                    ext = self.extension(c)
                    c.name = self.name + (ext if ext != "" else "")
                    yield Config(**c)
            else:
                ext = self.extension(config)
                config.name = self.name + (ext if ext != "" else "")
                yield Config(**config)

    def copy(self):
        return self.__class__(**self._state.copy())

    def extension(self, config):
        ext = ""
        for k, v in self._state.items():
            if isinstance(v, list) and all(
                [issubclass(vv.__class__, Config) for vv in v]
            ):
                ext += Config.separator + config[k].name
            elif issubclass(v.__class__, Config):
                ext += self[k].extension(config[k])
        return ext

    def summarize_subconfig(self, config: Config):
        values = {}
        for k, v in self._state.items():
            if self.is_sequence(k) and not issubclass(v[0].__class__, Config):
                values[k] = config._state[k]
            elif self.is_sequence(k) and issubclass(v[0].__class__, Config):
                for vv in v:
                    if vv.name != config._state[k].name:
                        continue
                    values.update(
                        {
                            k + self.separator + _k: _v
                            for _k, _v in vv.summarize_subconfig(
                                config._state[k]
                            ).items()
                        }
                    )
                    break
            elif issubclass(v.__class__, Config):
                values.update(
                    {
                        k + self.separator + _k: _v
                        for _k, _v in v.summarize_subconfig(config._state[k]).items()
                    }
                )
        return values
