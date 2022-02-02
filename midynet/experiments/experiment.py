import os
import numpy as np
import pathlib
import random
import time
import typing

from dataclasses import dataclass, field
from _midynet import utility
from midynet.config import Config, MetricsFactory
from midynet.metrics import Metrics
from midynet.util import (
    LoggerDict,
    Verbose,
    delete_path,
)

__all__ = ["Experiment"]


@dataclass
class Experiment:
    config: Config = field(repr=False, init=True)
    verbose: Verbose = field(default_factory=Verbose, init=True)
    loggers: LoggerDict = field(
        repr=False, default_factory=LoggerDict, init=True
    )
    name: str = field(default="exp", init=False)
    path: pathlib.Path = field(default_factory=pathlib.Path, init=False)
    config_filename: str = field(
        repr=False, default="config.pickle", init=False
    )
    log_filename: str = field(repr=False, default="log.json", init=False)
    seed: int = field(
        repr=False, default_factory=lambda: int(time.time()), init=False
    )
    metrics: typing.Dict[str, Metrics] = field(
        repr=False, default_factory=dict, init=False
    )

    def __post_init__(self):
        self.name = self.config.get_value("name", "exp")
        self.path = self.config.get_value("path", "./")

        if isinstance(self.verbose, int):
            if self.verbose == 1 or self.verbose == 2:
                self.verbose = Verbose(
                    filename=self.path / "verbose", verbose_type=self.verbose
                )
            else:
                self.verbose = Verbose(verbose_type=self.verbose)
        elif not isinstance(self.verbose, Verbose):
            message = (
                f"Invalid type `{type(self.verbose)}` for verbose,"
                + "expect `[int, Verbose]`."
            )
            raise TypeError(message)

        if isinstance(self.loggers, dict):
            self.loggers = LoggerDict(**self.loggers)

        self.__default_protocol__ = [
            "compute_metrics",
            "save",
        ]

    # Run command
    def run(self, protocol=None, clean=True):
        protocol = self.__default_protocol__ if protocol is None else protocol

        self.begin(clean)
        for t in protocol:
            if t in self.__default_protocol__:
                f = getattr(self, t)
                f()
            else:
                raise ValueError(
                    f"{t} is an invalid task,"
                    + f"possible tasks are `{self.__default_protocol__}`"
                )

        self.end()

    def begin(self, clean=True):
        self.path.mkdir(exist_ok=True, parents=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        utility.seed(self.seed)
        if clean:
            self.clean()

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
        self.config.save(self.path / self.config_filename)
        self.loggers.save(self.path / self.log_filename)
        for k, m in self.metrics.items():
            m.save(pathlib.Path(self.path) / f"{k}.pickle")

    def load(self):
        self.metrics = MetricsFactory.build(self.config)
        for k in self.config.metrics.metrics_names:
            self.metrics[k].load(pathlib.Path(self.path) / f"{k}.pickle")
        self.loggers.load(self.path / self.log_filename)

    def clean(self):
        for p in self.path.iterdir():
            delete_path(p)

    @classmethod
    def load_from_file(cls, path: pathlib.Path):
        path = (
            pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        )
        abs_path = path.resolve()
        config = Config.load(abs_path)
        config.set_value("path", abs_path.parents[0])
        exp = cls(config=config)
        exp.load()
        return exp

    @classmethod
    def merge(
        cls,
        name: str,
        path: pathlib.Path = ".",
        destination: pathlib.Path = None,
        prefix="",
        prohibited=None,
        config_filename="config.pickle",
    ):
        path = pathlib.Path(path) if isinstance(path, str) else path
        destination = (
            path / name if destination is None else destination / name
        )
        prohibited = [] if prohibited is None else prohibited
        others = []
        config = None
        for local, subpaths, files in os.walk(path):
            if local in prohibited:
                continue
            for f in files:

                if f == config_filename:
                    c = Config.load(pathlib.Path(local) / f)
                    others.append(cls.load_from_file(pathlib.Path(local) / f))
                    if config is None:
                        config = c
                        config.set_value("name", name)
                    else:
                        config.merge_with(c)
        config.set_value("path", destination)
        exp = cls(config)
        return exp


if __name__ == "__main__":
    pass
