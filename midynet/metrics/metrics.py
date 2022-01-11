import h5py
import numpy as np
import typing
from collections import defaultdict
from dataclasses import dataclass, field


from midynet.config import Config
from midynet.util import Verbose


@dataclass
class Metrics:
    config: Config = field(repr=False, default_factory=Config)
    verbose: Verbose = field(repr=False, default_factory=Verbose)
    data: typing.Dict[str, typing.Dict[str, np.ndarray]] = field(
        repr=False, default_factory=dict
    )
    num_procs: int = field(repr=True, default=1)

    def set_up(self, experiment):
        self.config = experiment.config
        self.num_procs = experiment.config.get_value("num_procs", 1)

    def tear_down(self, experiment):
        return

    def eval(self, config: Config) -> typing.Dict[str, float]:
        raise NotImplementedError()

    def compute(self, experiment, verbose=Verbose()) -> None:
        self.verbose = verbose
        self.set_up(experiment)

        pb = self.verbose.init_progress(self.__class__.__name__, total=len(self.config))
        raw_data = defaultdict(lambda: defaultdict(list))
        for c in self.config.generate_sequence():
            val = self.eval(c)
            for k, v in val.items():
                raw_data[c.name][k].append(v)
            if pb is not None:
                pb.update()
            self.verbose.update_progress()
        self.verbose.end_progress()
        self.data = self.format_data(raw_data)
        self.tear_down(experiment)

    def format_data(
        self, data: typing.Dict[str, typing.Dict[str, list]]
    ) -> typing.Dict[str, typing.Dict[str, np.array]]:
        formatted_data = {}
        for name, data_in_name in data.items():
            formatted_data[name] = {}
            for key, value in data_in_name.items():
                formatted_data[name][key] = np.zeros(
                    [len(v) for v in self.config.scanned_values[name].values()]
                )

                for i, c in enumerate(self.config.generate_sequence(only=name)):
                    index = [
                        np.where(
                            c.get_value(k)
                            == np.array(self.config.scanned_values[name][k])
                        )[0]
                        for k in self.config.scanned_keys[name]
                    ]
                    formatted_data[name][key][tuple(index)] = value[i]
        return formatted_data

    def unformat_data(self, data):
        unformatted_data = {}
        for name, data_in_name in data.items():
            unformatted_data[name] = {}

            for key, values in data_in_name.items():
                unformatted_data[name][key] = np.zeros(values.size)

                for i, c in enumerate(self.config.generate_sequence(only=name)):
                    index = [
                        np.where(
                            c.get_value(k)
                            == np.array(self.config.scanned_values[name][k])
                        )[0]
                        for k in self.config.scanned_keys[name]
                    ]
                    unformatted_data[name][key][i] = values[tuple(index)]
        return unformatted_data

    def save(self, h5file, name=None):
        if not isinstance(h5file, (h5py.File, h5py.Group)):
            raise ValueError("Dataset file format must be HDF5.")

        for name, data in self.data.items():
            for k, v in data.items():
                path = name + "/" + str(k)
                if path in h5file:
                    del h5file[path]
                h5file.create_dataset(path, data=v)

    def load(self, h5file, name=None):
        if not isinstance(h5file, (h5py.File, h5py.Group)):
            raise ValueError("Dataset file format must be HDF5.")

        name = name or self.__class__.__name__

        if name in h5file:
            self.data = self.read_h5_recursively(h5file[name])

    def read_h5_recursively(self, h5file, prefix=""):
        h5_dict = {}
        for key in h5file:
            item = h5file[key]
            if prefix == "":
                path = f"{key}"
            else:
                path = f"{prefix}/{key}"

            if isinstance(item, h5py.Dataset):
                h5_dict[path] = item[...]
            elif isinstance(item, h5py.Group):
                d = self.read_h5_recursively(item, path)
                h5_dict.update(d)
            else:
                raise ValueError()
        return h5_dict


class CustomMetrics(Metrics):
    def set_up(self, experiment):
        return


if __name__ == "__main__":
    pass
