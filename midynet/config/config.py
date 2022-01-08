import itertools
import typing

from .parameter import Parameter


class Config:
    name: str = "Config"
    separator: str = "/"

    def __init__(self, **kwargs):
        self.__parameters = {}
        for k, v in kwargs.items():
            self.insert(k, v)

    def keys(self, recursively=False):
        if recursively:
            return self.dict_copy(recursively=True).keys()
        return self.__parameters.keys()

    def values(self, recursively=False):
        if recursively:
            return self.dict_copy(recursively=True).values()
        return self.__parameters.values()

    def items(self, recursively=False):
        if recursively:
            return self.dict_copy(recursively=True).items()
        return self.__parameters.items()

    def insert(self, key: str, value: typing.Any):
        value = value.value if issubclass(type(value), Parameter) else value
        self.__parameters[key] = Parameter(name=key, value=value)
        self.__parameters[key].isConfig = issubclass(type(value), Config)

    def erase(self, key: str):
        self.__parameters.pop(key)

    def has_sequence(self):
        for v in self.values():
            if v.is_sequenced():
                return True
        return False

    def is_equivalent(self, other):
        for k, p in self.dict_copy(recursively=True).items():
            pp = other.get(k)
            if p.isConfig and not p.is_equivalent(pp):
                return False
            elif p.is_unique() and p.value != pp.value:
                return False
        return True

    def is_subconfig(self, other):
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
        s = self.name
        s += "("
        for k, v in self.items():
            if v.isConfig:
                s += f"{k}={v.value.name}, "
            else:
                s += f"{k}={v.value}, "
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
        if self.__parameters[key].isConfig and len(path) > 1:
            return self.__parameters[key].value.get(self.separator.join(path[1:]))
        else:
            return self.__parameters[key]

    def set(self, key: str, value: typing.Any) -> None:
        path = key.split(self.separator)
        key = path[0]
        if self.__parameters[key].isConfig and len(path) > 1:
            self.__parameters[key].set(self.separator.join(path[1:]), value)
        else:
            self.__parameters[key].set(value)

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

    def format(self, prefix=""):
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

    def generate_sequence(self):
        keys_to_scan = []
        values_to_scan = []
        for k, v in self.items():
            if v.isConfig and v.value.has_sequence():
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

    def merge(self) -> None:
        return


if __name__ == "__main__":
    pass
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
