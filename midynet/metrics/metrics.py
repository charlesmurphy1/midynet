import h5py
import numpy as np
import typing
import pathlib
import pickle
from collections import defaultdict
from dataclasses import dataclass, field


from midynet.config import Config
from midynet.util import Verbose

__all__ = ["Metrics"]


@dataclass
class Metrics:
    config: Config = field(repr=False, default_factory=Config)
    data: typing.Dict[str, typing.Dict[str, np.ndarray]] = field(
        repr=False, default_factory=dict, init=False
    )

    def set_up(self):
        return

    def tear_down(self):
        return

    def eval(self, config: Config) -> typing.Dict[str, float]:
        raise NotImplementedError()

    def compute(self, verbose=Verbose()) -> None:
        self.set_up()

        pb = verbose.init_progress(self.__class__.__name__, total=len(self.config))
        raw_data = defaultdict(lambda: defaultdict(list))
        for c in self.config.generate_sequence():
            val = self.eval(c)
            for k, v in val.items():
                raw_data[c.name][k].append(v)
            if pb is not None:
                pb.update()
            verbose.update_progress()
        verbose.end_progress()
        self.data = self.format(raw_data)
        self.tear_down()

    def format(self, data: np.ndarray) -> np.array:
        formatted_data = {}
        for name, data_in_name in data.items():
            formatted_data[name] = {}
            for key, value in data_in_name.items():
                formatted_data[name][key] = np.empty(
                    [len(v) for v in self.config.scanned_values[name].values()]
                )
                formatted_data[name][key][:] = np.nan

                for i, c in enumerate(self.config.generate_sequence(only=name)):
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

                for i, c in enumerate(self.config.generate_sequence(only=name)):
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

    def merge(self, other):
        if not isinstance(other, self.__class__):
            message = (
                f"Cannot merge since metrics of type {type(other)} "
                + f"is different from {self.__class__}."
            )
            raise TypeError(message)
        self_flat = self.flatten(self.data)
        other_flat = other.flatten(other.data)

        merged_config = self.config.deepcopy()
        merged_config.merge(other.config)
        merged_flat = {}

        for name, data_in_name in self_flat.items():
            # getting exclusive data from self
            if name not in other_flat:
                merged_flat[name] = data_in_name.copy()
                continue

        for name, data_in_name in other_flat.items():
            # getting exclusive data from other
            if name not in self_flat:
                merged_flat[name] = data_in_name.copy()
                continue
            merged_flat[name] = {}
            # getting shared data from other
            for key in data_in_name.keys():
                merged_flat[name][key] = []
                for i, c in enumerate(merged_config.generate_sequence(only=name)):
                    if self.config.is_subconfig(c):
                        # index = self.get_config_flat_index(c)
                        # merged_flat[name][key].append(self_flat[name][key][index])
                        merged_flat[name][key].append(np.nan)
                    elif other.config.is_subconfig(c):
                        # index = other.get_config_flat_index(c)
                        # merged_flat[name][key].append(other_flat[name][key][index])
                        merged_flat[name][key].append(np.nan)
                    else:
                        merged_flat[name][key].append(np.nan)
        self.config = merged_config
        # self.data = self.format(merged_flat)

    def get_config_indices(self, local_config):
        indices = []
        for k in self.config.scanned_keys[local_config.name]:
            value = local_config.get_value(k)
            all_values = np.array(self.config.scanned_values[local_config.name][k])
            i = np.where(value == all_values)[0]
            if i.size == 0:
                message = "Cannot get indices, config not found."
                raise ValueError(message)
            indices.append(i[0])
        return tuple(indices)

    def get_config_flat_index(self, local_config):
        for i, c in enumerate(self.config.generate_sequence(only=local_config.name)):
            if local_config.is_equivalent(c):
                return i
        message = "Cannot get flat index, config not found."
        raise ValueError(message)


class CustomMetrics(Metrics):
    def set_up(self, experiment):
        return


if __name__ == "__main__":
    pass
