import numpy as np
import os
import pathlib
import pickle
import random
import time
import typing

# import shutil
# import zipfile

from dataclasses import dataclass, field

from midynet.config import Config, MetricsFactory
from midynet.metrics import Metrics
from midynet.util import Verbose, LoggerDict, TimeLogger, MemoryLogger, delete_path
from _midynet import utility

__all__ = ["Experiment"]


@dataclass
class Experiment:
    config: Config = field(repr=False, init=True)
    verbose: Verbose = field(default_factory=Verbose, init=True)
    loggers: LoggerDict = field(repr=False, default_factory=LoggerDict, init=True)
    name: str = field(default="exp", init=False)
    path: pathlib.Path = field(default_factory=pathlib.Path, init=False)
    config_filename: str = field(repr=False, default="config.pickle", init=False)
    log_filename: str = field(repr=False, default="log.pickle", init=False)
    seed: int = field(repr=False, default_factory=lambda: int(time.time()), init=False)
    metrics: typing.Dict[str, Metrics] = field(
        repr=False, default_factory=dict, init=False
    )

    def __post_init__(self):
        self.name = self.config.get_value("name", "exp")
        self.path = self.config.get_value("path", "./")
        self.path.mkdir(exist_ok=True, parents=True)

        if isinstance(self.verbose, int):
            if self.verbose == 1 or self.verbose == 2:
                self.verbose = Verbose(
                    filename=self.path / "verbose", verbose_type=self.verbose
                )
            else:
                self.verbose = Verbose(verbose_type=self.verbose)
        elif not isinstance(self.verbose, Verbose):
            message = f"Invalid type `{type(self.verbose)}` for verbose, expect `[int, Verbose]`."
            raise TypeError(message)

        if isinstance(self.loggers, dict):
            self.loggers = LoggerDict(**self.loggers)

        random.seed(self.seed)
        np.random.seed(self.seed)
        utility.seed(self.seed)

        self.__default_protocol__ = [
            "compute_metrics",
            "save",
            "zip",
        ]

    # Run command
    def run(self, protocol=None, clean=True):
        protocol = self.__default_protocol__ if protocol is None else protocol
        if clean:
            self.clean()

        self.begin()
        for t in protocol:
            if t in self.__default_protocol__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task, possible tasks are `{self.__default_protocol__}`"
                )

        self.end()

    def begin(self):
        self.loggers.on_task_begin()
        self.verbose(f"---Experiment {self.name}---")
        if "time" in self.loggers.keys():
            begin = self.loggers["time"].log["begin"]
            self.verbose(f"Current time: {begin}")
        self.verbose(self.config.format())

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

        self.metrics = MetricsFactory.build(self.config)
        for k in self.config.metrics.metrics_names:
            self.loggers.on_task_update("metrics")
            self.metrics[k].compute(verbose=self.verbose)
            if save:
                self.metrics[k].save(pathlib.Path(self.path) / f"{k}.pickle")

    def save(self):
        self.config.save(self.path_data / self.config_filename)
        self.loggers.save(self.path_data / self.log_filename)
        for k, m in self.metrics.items():
            m.save(pathlib.Path(self.path) / f"{k}.pickle")

    def load(self):
        for k in self.config.metrics.metrics_names:
            self.metrics[k] = MetricsFactory.build(k)
            self.metrics[k].load(pathlib.Path(self.path) / f"{k}.pickle")
        self.loggers.load(self.path / self.log_filename)

    @classmethod
    def load_from_file(cls, name: str, path: pathlib.Path):
        config = Config.load(path)
        exp = cls(name, config=config, path=path)
        exp.load()
        return exp

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
        others = []
        config = None
        counter = 0
        for local, subpaths, files in os.walk(path):
            if local in prohibited:
                continue
            for f in files:

                if f == config_filename:
                    c = Config.load(pathlib.Path(local) / f)
                    others.append(
                        cls.load_from_file(f"{name}_{counter}", pathlib.Path(local))
                    )
                    if config is None:
                        config = c
                        config.set_value("name", name)
                    else:
                        config.merge_with(c)

        exp = cls(name, config, destination, config_filename=config_filename)
        for k in exp.config.metrics.names:
            exp.metrics[k] = MetricsFactory.build(k)
            for other in others:
                exp.metrics[k].merge_with(other.metrics[k])
        return exp


if __name__ == "__main__":
    pass
