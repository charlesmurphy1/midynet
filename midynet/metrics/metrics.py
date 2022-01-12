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
    verbose: Verbose = field(repr=False, default_factory=Verbose)
    data: typing.Dict[str, typing.Dict[str, np.ndarray]] = field(
        repr=False, default_factory=dict
    )

    def set_up(self, experiment):
        self.config = experiment.config

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

    def unformat_data(
        self, data: typing.Dict[str, typing.Dict[str, np.array]]
    ) -> typing.Dict[str, typing.Dict[str, np.array]]:
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

    def save(self, path: pathlib.Path = "metrics.pickle"):
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open("wb") as f:
            pickle.dump(self.data, f)

    def load(self, path: pathlib.Path):
        path = pathlib.Path(path) if isinstance(path, str) else path
        with path.open("rb") as f:
            self.data = pickle.load(f)


class CustomMetrics(Metrics):
    def set_up(self, experiment):
        return


if __name__ == "__main__":
    pass
