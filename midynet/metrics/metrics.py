import pathlib
import pickle
import pandas as pd
import numpy as np
import datetime

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
        for log in self.logs:
            log.setup(total=len(configs))
        for config in (
            configs.to_sequence()
            if issubclass(configs.__class__, Config)
            else [configs]
        ):
            res = self.eval(config)
            res.update(configs.summarize_subconfig(config))
            for k, v in res.items():
                raw_data[config.name][k].append(v)
            for log in self.logs:
                log.update(logger)
        self.data = {k: pd.DataFrame(v) for k, v in raw_data.items()}

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
