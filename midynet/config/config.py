from __future__ import annotations

from itertools import product
from typing import Iterable, List
from pyhectiqlab import Config as Config
from copy import deepcopy


class ParameterSequence(list):
    pass


class MetaConfig(Config):
    separator: str = "."

    def __init__(self, name="config", **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def __len__(self):
        return len(list(self.to_sequence()))

    def sequenced_items(self):
        for k, v in self._state.items():
            if isinstance(v, ParameterSequence):
                yield k, v
            elif issubclass(v.__class__, MetaConfig) and v.is_sequenced():
                yield k, v.to_sequence()

    @property
    def dict(self):
        """A property that converts a config to a dict. Supports nested Config."""
        d = {}
        for key, item in self._state.items():
            if issubclass(item.__class__, Config):
                d[key] = item.dict
            elif issubclass(item.__class__, list):
                d[key] = [
                    i.dict if issubclass(i.__class__, Config) else i
                    for i in item
                ]
            else:
                d[key] = item
        return d

    @staticmethod
    def from_dict(data):
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

        config = MetaConfig()
        for key in data:
            if isinstance(data[key], dict):
                config[key] = MetaConfig.from_dict(data[key])
            elif issubclass(data[key].__class__, list):
                config[key] = [MetaConfig.from_dict(d) for d in data[key]]
            else:
                config[key] = data[key]
        return config

    @classmethod
    def auto(cls, config: str or MetaConfig, *args, **kwargs):
        if config in cls.__dict__:
            return getattr(cls, config)(*args, **kwargs)
        elif isinstance(config, cls):
            return config
        else:
            t = config if isinstance(config, str) else type(config)
            message = f"Invalid config type `{t}` for auto build of object `{cls.__name__}`."
            raise TypeError(message)

    def is_sequenced(self):
        for k, v in self._state.items():
            if isinstance(v, ParameterSequence):
                return True
            elif isinstance(v, MetaConfig) and v.is_sequenced():
                return True
        return False

    def to_sequence(self):
        if not self.is_sequenced():
            yield self
            return

        keys_to_scan = []
        values_to_scan = []

        for k, v in self.sequenced_items():
            keys_to_scan.append(k)
            values_to_scan.append(v)
        for values in product(*values_to_scan):
            config = self.copy()
            for k, v in zip(keys_to_scan, values):
                setattr(config, k, v)
            if config.is_sequenced():
                for c in config.to_sequence():
                    ext = self.extension(c)
                    c.name = self.name + (ext if ext != "" else "")
                    yield c
            else:
                ext = self.extension(config)
                config.name = self.name + (ext if ext != "" else "")
                yield config

    def copy(self):
        return self.__class__(**self._state)

    def extension(self, config):
        ext = ""
        for k, v in self._state.items():
            if isinstance(v, list) and all(
                [issubclass(vv.__class__, Config) for vv in v]
            ):
                ext += MetaConfig.separator + config[k].name
            elif issubclass(v.__class__, MetaConfig):
                ext += self[k].extension(config[k])
        return ext
