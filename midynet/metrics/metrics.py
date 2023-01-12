import pathlib
import pandas as pd
import logging
import sys

from tempfile import mktemp
from typing import Dict, Optional
from logging import Logger
from collections import defaultdict

from midynet.config import Config
from midynet.metrics.logger import ProgressLog, MemoryLog

__all__ = ("Metrics",)


class Metrics:
    shortname = "metrics"
    keys = []

    def __init__(self, logs="all"):
        self.data = {}
        if logs == "all":
            self.logs = [
                ProgressLog(),
                MemoryLog(),
            ]
        elif logs is None:
            self.logs = []
        else:
            self.logs = logs

    def eval(self, config: Config) -> Dict[str, float]:
        raise NotImplementedError()

    def compute(self, configs: Config, logger: Logger = None) -> None:
        for log in self.logs:
            log.setup(total=len(configs))
        if logger == "stdout":
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        self.data = defaultdict(pd.DataFrame)
        if logger is not None:
            logger.info(f"---Computing {self.__class__.__name__}---")
        for config in (
            configs.to_sequence()
            if issubclass(configs.__class__, Config)
            else [configs]
        ):
            raw = pd.DataFrame(self.eval(config))
            for k, v in configs.summarize_subconfig(config).items():
                raw[k] = v

            self.data[config.name] = pd.concat(
                [self.data[config.name], raw], ignore_index=True
            )
            for log in self.logs:
                log.update(logger)
        self.data = dict(self.data)
        if len(self.data) == 1:
            self.data = next(iter(self.data.values()))

    def to_pickle(
        self, path: Optional[str or pathlib.Path] = None, **kwargs
    ) -> str:
        path = (
            pathlib.Path(mktemp()) if path is None else pathlib.Path(path)
        ) / f"{self.shortname}.pkl"
        pd.to_pickle(self.data, path, **kwargs)
        return str(path)

    def read_pickle(self, path: str or pathlib.Path, **kwargs):
        self.data = pd.read_pickle(path, **kwargs)


if __name__ == "__main__":
    pass
