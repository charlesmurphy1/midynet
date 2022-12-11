import pathlib
import pandas as pd
import logging
import sys

from tempfile import mktemp
from typing import Dict, Optional, Tuple
from logging import Logger
from collections import defaultdict
from pyhectiqlab import Run

from midynet.config import Config
from midynet.metrics.logger import ProgressLog, MemoryLog

__all__ = ("Metrics",)


class Metrics:
    def __init__(self, name, logs="all"):
        self.name = name
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
        raw_data = defaultdict(lambda: defaultdict(list))
        params = defaultdict(lambda: defaultdict(list))
        for log in self.logs:
            log.setup(total=len(configs))
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.info(f"---Computing {self.__class__.__name__}---")
        for config in (
            configs.to_sequence()
            if issubclass(configs.__class__, Config)
            else [configs]
        ):
            for k, v in self.eval(config).items():
                raw_data[config.name][k].append(v)
            for k, v in configs.summarize_subconfig(config).items():
                params[config.name][k].append(v)
            for log in self.logs:
                log.update(logger)
        self.data = {
            k: dict(
                metrics=pd.DataFrame(raw_data[k]),
                params=pd.DataFrame(params[k]),
            )
            for k in raw_data.keys()
        }

    def to_pickle(
        self, path: Optional[str or pathlib.Path] = None, **kwargs
    ) -> str:
        path = (
            pathlib.Path(mktemp()) if path is None else pathlib.Path(path)
        ) / f"{self.name}.pkl"
        pd.to_pickle(self.data, path, **kwargs)
        return str(path)

    def read_pickle(self, path: str or pathlib.Path, **kwargs):
        self.data = pd.read_pickle(path, **kwargs)


if __name__ == "__main__":
    pass
