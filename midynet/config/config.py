import numpy as np
import typing
from itertools import product

from .parameter import Parameter


class Config:
    name: str = "Config"
    separator: str = "/"

    def __init__(self, **kwargs):
        self.__parameters: dict = {}
        for k, v in kwargs.items():
            self.insert(k, v)

    def keys(self, recursively=False):
        if recursively:
            return self.recursive_copy().keys()
        return self.__parameters.keys()

    def values(self, recursively=False):
        if recursively:
            return self.recursive_copy().keys()
        return self.__parameters.values()

    def items(self, recursively=False):
        return self.__parameters.items()

    def insert(self, key: str, value: typing.Any):
        self.__parameters[key] = Parameter(name=key, value=value)
        self.__parameters[key].isConfig = issubclass(type(value), Config)

    def __str__(self) -> str:
        s = self.name
        s += "("
        for k, v in self.items():
            if v.isConfig:
                s += f"{k}={v.value.name}, "
            else:
                s += f"{k}={v.value}"
        s = s[:-2]
        s += ")"
        return s

    def __repr__(self) -> str:
        return "Config()"

    def __contains__(self, key) -> bool:
        return key in self.keys()

    def __setitem__(self, key: str, value: typing.Any) -> None:

        if not isinstance(value, self.__parameters[key].datatype):
            message = f"type `{type(value)}` of key `{key}` is invalid, expected type `{self.__parameters[key].datatype}`."
            raise TypeError(message)

        if key in self:
            self.__parameters[key].value = value
        else:
            self.insert(key, value)

    def __getitem__(self, key: str) -> Parameter:
        if key not in self:
            message = f"key `{key}` has not been found."
            raise LookupError(message)
        return self.__parameters[key]

    def get(self, key: str) -> Parameter:
        path = key.split(self.separator)
        key = path[0]
        if self[key].isConfig and len(path) > 1:
            return self[key].value.get(self.separator.join(path[1:]))
        else:
            return self[key]

    def set(self, key: str, value: typing.Any) -> None:
        path = key.split(self.separator)
        key = path[0]

        if self[key].isConfig:
            self[key].set(self.separator.join(path[1:]), value, self.separator)
        else:
            self[key] = value

    def copy(self, recursively=False, prefix="") -> typing.Dict[str, Parameter]:
        copy = {}

        for k, v in self.items():
            if v.isConfig and recursively:
                copy.update(
                    v.value.copy(
                        recursively=recursively,
                        prefix=prefix + k + self.separator,
                    )
                )
            else:
                copy[prefix + k] = v

        return copy


if __name__ == "__main__":
    pass
    # def generate_sequence(self) -> typing.Iterator[Config]:
    #     return

    # def get_recursive_config(self):

    # def __gt__(self, other):
    #     for k, v in self.state_dict.items():
    #         vv = other[k]
    #         if (isinstance(v, list) and vv in v) or k == "seed" or k == "path_to_data":
    #             pass
    #         elif isinstance(v, list) and vv not in v:
    #             return False
    #         elif isinstance(v, list) and isinstance(vv, list):
    #             s, ss = set(v), set(vv)
    #             if len(s.difference(ss)) > 0:
    #                 return False
    #         elif not isinstance(v, list) and isinstance(vv, list):
    #             return False
    #         elif not isinstance(v, list) and not isinstance(vv, list) and v != vv:
    #             return False
    #     return True
    #
    # def __eq__(self, other):
    #     for k, v in self.items():
    #         if k == "seed" or k == "path_to_data":
    #             pass
    #         elif k not in other:
    #             return False
    #         else:
    #             is_equal = other[k] == v
    #             if isinstance(is_equal, bool) and not all(is_equal):
    #                 return False
    #             elif isinstance(is_equal, np.ndarray) and not np.all(is_equal):
    #                 return False
    #     return True
    #
    # def __setitem__(self, key, val):
    #     key = key.split("/")
    #     if len(key) == 1:
    #         setattr(self, key[0], val)
    #     else:
    #         config = getattr(self, key[0])
    #         key = "/".join(key[1:])
    #         config[key] = val
    #
    # def __getitem__(self, key):
    #     key = key.split("/")
    #     if len(key) == 1:
    #         return getattr(self, key[0])
    #     else:
    #         config = getattr(self, key[0])
    #         key = "/".join(key[1:])
    #         return config[key]
    #
    # def get(self, key, default=None):
    #     return default if key not in self else self[key]
    #
    # def to_string(self, prefix=""):
    #     string = ""
    #     for k, v in self.__dict__.items():
    #         if issubclass(v.__class__, Config):
    #             string += prefix + f"{k}:\n"
    #             string += "{0}\n".format(v.to_string(prefix=prefix + "\t"))
    #         else:
    #             string += prefix + f"{k}: {v.__str__()}\n"
    #     return string
    #
    # def get_state_dict(self):
    #     state_dict = {}
    #     for k, v in self.__dict__.items():
    #         if k != "_state_dict":
    #             if issubclass(v.__class__, Config):
    #                 v_dict = v.state_dict
    #                 for kk, vv in v_dict.items():
    #                     state_dict[k + "/" + kk] = vv
    #             else:
    #                 state_dict[k] = v
    #     return state_dict
    #
    # @property
    # def state_dict(self):
    #     return self.get_state_dict()
    #
    # def has_list(self):
    #     for k, v in self.state_dict.items():
    #         if isinstance(v, list):
    #             return True
    #     return False
    #
    # def merge(self, config):
    #     for k, v in config.__dict__.items():
    #         self.__dict__[k] = v
    #
    # def copy(self):
    #     config_copy = self.__class__()
    #     for k, v in self.__dict__.items():
    #         # if hasattr(v, "copy") and callable(getattr(v, "copy")):
    #         if issubclass(v.__class__, Config) or isinstance(v, (np.ndarray, list)):
    #             setattr(config_copy, k, v.copy())
    #         else:
    #             setattr(config_copy, k, v)
    #     return config_copy
    #


# class ConfigArray:
#     def __init__(self, config):
#         self.config = config
#
#     @property
#     def config(self):
#         return self._config
#
#     @config.setter
#     def config(self, config):
#         self._config = config
#         self.reset()
#
#     def __str__(self):
#         return self.config.to_string()
#
#     def keys(self):
#         return self.config.keys()
#
#     def values(self):
#         return self.config.values()
#
#     def items(self):
#         return self.config.items()
#
#     def to_string(self, prefix=""):
#         string = self.config.to_string()
#         for k, v in self.__dict__.items():
#             if k == "config":
#                 pass
#             elif issubclass(v.__class__, Config):
#                 string += prefix + f"{k}:\n"
#                 string += "{0}\n".format(v.to_string(prefix=prefix + "\t"))
#             else:
#                 string += prefix + f"{k}: {v.__str__()}\n"
#         return string
#
#     def reset(self):
#         self.labels_to_scan = []
#         self.values_to_scan = []
#
#         for k, v in self._config.state_dict.items():
#             if isinstance(v, list):
#                 self.labels_to_scan.append(k)
#                 self.values_to_scan.append(v)
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
#     def __len__(self):
#         return len(list(product(*self.values_to_scan)))
#
#     def __iter__(self):
#         for v in product(*self.values_to_scan):
#             local_config = self.config.copy()
#             for k, vv in zip(self.labels_to_scan, v):
#                 local_config[k] = vv
#             local_config.scan = {k: vv for k, vv in zip(self.labels_to_scan, v)}
#             yield local_config
#
#     @property
#     def scans(self):
#         return np.array([v for v in product(*self.values_to_scan)])
#
#     def get_scan_index(self, c):
#         scan = np.array([c.scan[l] for l in self.labels_to_scan])
#         return np.where(np.all(self.scans == scan, axis=1))[0][0]
#
#     @property
#     def state_dict(self):
#         return self.config.get_state_dict()
#
#     def batch(self, size):
#         l = len(self)
#         for i in range(0, l, size):
#             yield self[i : min(i + size, l)]
