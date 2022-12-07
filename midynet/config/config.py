from __future__ import annotations

from itertools import product
from typing import List
from pyhectiqlab import Config as BaseConfig
from copy import deepcopy


class ParameterSequence(list):
    pass


class Config(BaseConfig):
    separator: str = "."

    def __init__(self, name="config", **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def get(self, key, default=None):
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

    def __len__(self):
        return len(list(self.to_sequence()))

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

        config = Config()
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
        return config

    @classmethod
    def auto(cls, config: str or Config or List[Config], *args, **kwargs):
        configs = [config] if not isinstance(config, list) else config
        res = []
        for config in configs:
            if config in cls.__dict__:
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
        return ParameterSequence(res)

    def is_sequenced(self):
        for k, v in self._state.items():
            if isinstance(v, ParameterSequence):
                return True
            elif isinstance(v, Config) and v.is_sequenced():
                return True
        return False

    def as_sequence(self, key):
        if isinstance(self._state[key], ParameterSequence):
            return
        if isinstance(self._state[key], (list, tuple, set)):
            self._state[key] = ParameterSequence(self._state[key])

    def to_sequence(self):
        if not self.is_sequenced():
            yield self
            return

        keys_to_scan = []
        values_to_scan = []

        for k, v in self._state.items():
            if isinstance(v, ParameterSequence):
                keys_to_scan.append(k)
                values_to_scan.append(v)
            elif issubclass(v.__class__, Config) and v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.to_sequence())
        for values in product(*values_to_scan):
            config = self.copy()
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
            if isinstance(v, ParameterSequence) and not issubclass(
                v[0].__class__, Config
            ):
                values[k] = config._state[k]
            elif isinstance(v, ParameterSequence) and issubclass(
                v[0].__class__, Config
            ):
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
                        for _k, _v in v.summarize_subconfig(
                            config._state[k]
                        ).items()
                    }
                )
        return values
