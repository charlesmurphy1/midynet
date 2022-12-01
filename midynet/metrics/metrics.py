import pathlib
import pickle
import typing
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

from pyhectiqlab import Config
from midynet.utility import Verbose, to_batch
from .multiprocess import NestablePool

__all__ = ("Metrics",)


@dataclass
class Metrics:
    config: Config = field(repr=False, default=None)
    data: typing.Dict[str, typing.Dict[str, np.ndarray]] = field(
        repr=False, default_factory=dict, init=False
    )
    counter: int = field(repr=False, default=0, init=False)
    raw_data: dict = field(repr=False, default_factory=dict)

    def set_up(self):
        self.counter = 0

    def tear_down(self):
        self.counter = 0

    def eval(self, config: Config) -> typing.Dict[str, float]:
        raise NotImplementedError()

    def compute(self, verbose=Verbose()) -> None:
        self.set_up()
        pb = verbose.init_progress(
            self.__class__.__name__, total=len(self.config)
        )
        raw_data = defaultdict(lambda: defaultdict(list))
        num_async_process = self.config.get_value("num_async_process", 1)
        for batch in to_batch(self.config.sequence(), num_async_process):
            pool = NestablePool(num_async_process)
            vals = pool.map(self.eval, batch)
            pool.close()
            pool.join()
            for val, c in zip(vals, batch):
                for k, v in val.items():
                    raw_data[c.name][k].append(v)
                if pb is not None:
                    pb.update()
                verbose.update_progress()
        verbose.end_progress()

        self.data = self.format(raw_data)
        self.tear_down()

    def format(self, data: dict) -> np.array:
        formatted_data = {}
        for name, data_in_name in data.items():
            formatted_data[name] = {}
            for key, value in data_in_name.items():
                formatted_data[name][key] = np.empty(
                    [
                        len(v)
                        for v in self.config.scanned_values()[name].values()
                    ]
                )

                if formatted_data[name][key].shape == ():
                    formatted_data[name][key] = formatted_data[name][
                        key
                    ].reshape(1)
                formatted_data[name][key][:] = np.nan
                for i, c in enumerate(self.config.named_sequence(name)):
                    index = self.get_config_indices(c)
                    formatted_data[name][key][index] = value[i]
        return formatted_data

    def flatten(
        self, data: typing.Dict[str, typing.Dict[str, np.array]]
    ) -> typing.Dict[str, typing.Dict[str, np.array]]:
        flat_data = {}
        for name, data_in_name in data.items():
            flat_data[name] = {}

            for key, values in data_in_name.items():
                flat_data[name][key] = np.zeros(values.size)

                for i, c in enumerate(self.config.named_sequence(name)):
                    index = self.get_config_indices(c)
                    flat_data[name][key][i] = values[index]
        return flat_data

    def save(self, path: pathlib.Path):
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open("wb") as f:
            pickle.dump(self.data, f)

    def load(self, path: pathlib.Path):
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open("rb") as f:
            self.data = pickle.load(f)

    def merge_with(self, other):

        if not isinstance(other, self.__class__):
            message = (
                f"Cannot merge since metrics of type {type(other)} "
                + f"is different from {self.__class__}."
            )
            raise TypeError(message)
        self_flat = self.flatten(self.data)
        other_flat = other.flatten(other.data)

        merged_config = self.config.deepcopy()
        merged_config.merge_with(other.config)
        merged_flat = defaultdict(lambda: defaultdict(list))

        for c in merged_config.sequence():
            name = merged_config.subname(c)
            self_name = self.config.subname(c)
            other_name = other.config.subname(c)
            for key in self.data[self_name].keys():
                if c.is_subconfig(self.config):
                    index = self.get_config_flat_index(c, name=self_name)
                    merged_flat[name][key].append(
                        self_flat[self_name][key][index]
                    )
                elif c.is_subconfig(other.config):
                    index = other.get_config_flat_index(c, name=other_name)
                    merged_flat[name][key].append(
                        other_flat[other_name][key][index]
                    )
                else:
                    merged_flat[name][key].append(np.nan)

        self.config = merged_config
        self.data = self.format(merged_flat)

    def get_config_indices(self, local_config, name=None):
        name = local_config.name if name is None else name
        indices = []
        for k in self.config.scanned_keys()[name]:
            value = local_config.get_value(k)
            all_values = np.array(self.config.scanned_values()[name][k])
            i = np.where(value == all_values)[0]
            if i.size == 0:
                message = "Cannot get indices, config not found."
                raise ValueError(message)
            indices.append(i[0])
        return tuple(indices)

    def get_config_flat_index(self, local_config, name=None):
        name = local_config.name if name is None else name
        h = hash(local_config)
        if h not in self.config.hash_dict()[name]:
            message = "Cannot get flat index, config not found."
            raise ValueError(message)
        return self.config.hash_dict()[name].index(h)


if __name__ == "__main__":
    pass
