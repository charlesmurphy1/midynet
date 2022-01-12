import h5py
import json
import numpy as np
import os
import pickle
import random
import shutil
import zipfile
import time

from dataclasses import dataclass, field

from midynet.config import Config, MetricsFactory
from midynet.util import Verbose, LoggerDict, TimeLogger, MemoryLogger, delete_path
from _midynet import utility


@dataclass
class Experiment:
    name: str
    config: Config
    path: pathlib.Path = field(default_factory=pathlib.Path)
    verbose: Verbose = field(default_factory=Verbose)
    config_filename: str = field(repr=False, default="config.pickle")
    log_filename: str = field(repr=False, default="log.pickle")
    seed: int = field(repr=False, default_factory=lambda: int(time.time()))
    loggers: LoggerDict = field(repr=False, default_factory=LoggerDict)
    metrics: Dict[str, Metrics] = field(repr=False, default_factory=dict)

    def __post_init__(self):
        self.path.mkdir(exist_ok=True)

        if isinstance(self.verbose, int):
            if verbose == 1 or verbose == 2:
                self.verbose = Verbose(filename=self.path / "verbose", vtype=verbose)
            else:
                self.verbose = Verbose(vtype=verbose)
        elif not isinstance(self.verbose, Verbose):
            message = (
                f"Invalid type `{type(verbose)}` for verbose, expect `[int, Verbose]`."
            )
            raise TypeError(message)

        random.seed(self.seed)
        np.random.seed(self.seed)
        utility.seed(self.seed)

        self.__tasks__ = [
            "load",
            "compute_metrics",
            "save",
            "zip",
        ]

    # Run command
    def run(self, tasks=None):
        tasks = tasks or self.__tasks__

        self.begin()
        for t in tasks:
            if t in self.__tasks__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.__tasks__}`"
                )

        self.end()

    def begin(self):
        self.loggers.on_task_begin()
        self.verbose(f"---Experiment {self.name}---")
        if "time" in self.loggers.keys():
            begin = self.loggers["time"].log["begin"]
            self.verbose(f"Current time: {begin}")
        self.verbose(self.config)

    def end(self):
        self.loggers.on_task_end()
        self.verbose(f"\n---Finished {self.name}---")
        if "time" in self.loggers.keys():
            end = self.loggers["time"].log["end"]
            t = self.loggers["time"].log["time"]
            self.verbose(f"Current time: {end}")
            self.verbose(f"Computation time: {t}\n")

    # All tasks
    def compute_metrics(self, save=True):
        self.verbose("\n---Computing metrics---")

        for k in self.config.metrics.names:
            self.metrics[k] = MetricsFactory.build(k)
            self.loggers.on_task_update("metrics")
            self.metrics[k].compute(self, verbose=self.verbose)
            if save:
                self.metrics[k].save(pathlib.Path(self.path) / f"{k}.pickle")

    def save(self):
        self.config.save(self.path_data / self.config_filename)
        self.loggers.save(self.path_data / self.log_filename)
        for k, m in self.metrics.items():
            m.save(pathlib.Path(self.path) / f"{k}.pickle")

    def load(self):
        for k in self.config.metrics.names:
            self.metrics[k] = MetricsFactory.build(k)
            self.metrics[k].load(pathlib.Path(self.path) / f"{k}.pickle")
        self.loggers.load(self.path / self.log_filename)

    # @classmethod
    # def from_file(cls, path_to_config, autoload=True):
    #     with open(path_to_config, "rb") as config_file:
    #         config = pickle.load(config_file)
    #     exp = cls(config)
    #     if autoload:
    #         exp.load()
    #     return exp
    #
    # @classmethod
    # def unzip(cls, path_to_zip, destination=None):
    #     zip = zipfile.ZipFile(path_to_zip, mode="r")
    #     path_to_data, _ = os.path.split(zip.namelist()[0])
    #     destination = destination or "."
    #     zip.extractall(path=destination)
    #     cls = cls.from_file(os.path.join(path_to_data, "config.pickle"))
    #     cls.path_to_data = path_to_data
    #     cls.load()
    #     shutil.rmtree(path_to_data)
    #     return cls

    def clean(self):
        for p in self.path.iterdir():
            delete_path(p)

    @classmethod
    def merge(
        cls,
        name: str,
        path: pathlib.Path = None,
        destination: pathlib.Path = None,
        prefix="",
        prohibited=None,
        config_filename="config.pickle",
    ):
        path = pathlib.Path() if path is None else path
        destination = pathlib.Path(name) if destination is None else destination / name
        prohibited = [] if prohibited is None else prohibited
        config = None
        for local, subpaths, files in os.walk(path):
            if local in prohibited:
                continue
            for f in files:

                if f == config_filename:
                    if config is None:
                        config = Config.load(pathlib.Path(local) / f)
                        config.set_value(name, config_name)
                    else:
                        config.merge(Config.load(pathlib.Path(local) / f))

        exp = cls(name, config, destionation, config_filename=config_filename)

        # configs = get_config_for_merger(
        #     location=location, prefix=prefix, prohibited=prohibited
        # )

        # merge_seed = int(time.time())
        exp = None
        # for path, c in configs.items():
        #     c.name = prefix
        #     c.seed = merge_seed
        #     c.path_to_data = path
        #     other = cls(c)
        #     other.load_metrics()
        #     if exp is None:
        #         exp = other
        #     else:
        #         exp = exp.merge_with(other)
        return exp

    # def merge_with(self, other):
    #     if issubclass(type(other), Experiment):
    #         pass
    #     elif issubclass(type(other), Config):
    #         other = Experiment(other)
    #         other.load()
    #     elif os.path.isfile(other):
    #         other = Experiment.from_file(other)
    #         other.load()
    #     else:
    #         raise TypeError()
    #     shared_config = self.get_shared_config(other)
    #     merged_experiment = Experiment(shared_config)
    #     for n, m in self.metrics.items():
    #         mm = other.metrics[n]
    #         data = {}
    #         v = m.unformat_data(m.data)
    #         vv = mm.unformat_data(mm.data)
    #
    #         for k in v.keys():
    #             data[k] = []
    #             for i, c in enumerate(merged_experiment.config_array):
    #                 if self.config > c and not (other.config > c):
    #                     index = self.config_array.get_scan_index(c)
    #                     data[k].append(v[k][index])
    #                 elif not (self.config > c) and other.config > c:
    #                     index = other.config_array.get_scan_index(c)
    #                     data[k].append(vv[k][index])
    #                 elif not (self.config > c) and not (other.config > c):
    #                     data[k].append(np.nan)
    #                 else:
    #                     print("ERROR", c.scan)
    #
    #         data = merged_experiment.metrics[n].format_data(data)
    #         merged_experiment.metrics[n].data = data
    #     return merged_experiment
    #
    # def get_shared_config(self, other):
    #     shared_config = self.config.copy()
    #     config = other.config
    #     for k, v in self.config.state_dict.items():
    #         vv = config[k]
    #         if k not in other.config.state_dict:
    #             msg = f"Key `{k}` was not found in other."
    #             raise RuntimeError(msg)
    #         if v == config[k] or k == "seed" or k == "path_to_data":
    #             pass
    #         else:
    #             if not isinstance(v, list):
    #                 v = [v]
    #             if not isinstance(vv, list):
    #                 vv = [vv]
    #             shared_config[k] = sorted(list(set(v + vv)))
    #     return shared_config


if __name__ == "__main__":
    pass
