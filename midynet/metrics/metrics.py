import pathlib
import pandas as pd
import logging
import sys
import os
import numpy as np
import time

from tempfile import mkdtemp
from typing import Dict, Optional
from logging import Logger
from collections import defaultdict

from midynet.config import Config
from midynet.metrics.logger import ProgressLog, MemoryLog
from .multiprocess import Expectation
from midynet.statistics import Statistics

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

    def compute(
        self,
        configs: Config,
        logger: Logger = None,
        resume: bool = True,
        save_path: str = None,
        patience: int = 5,
    ) -> None:
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

        data = defaultdict(pd.DataFrame)
        if save_path is not None and resume and os.path.exists(save_path):
            self.read_pickle(save_path)
            data.update(self.data)
        for i, config in enumerate(
            configs.to_sequence()
            if issubclass(configs.__class__, Config)
            else [configs]
        ):
            skip = False

            if resume and config.name in data:
                cond = pd.DataFrame()
                # params_is_matched = dict()
                for k, v in configs.summarize_subconfig(config).items():
                    cond[k] = data[config.name][k] == v
                skip = np.any(np.prod(cond.values, axis=-1))
            if not skip:
                raw = pd.DataFrame(self.eval(config))
                for k, v in configs.summarize_subconfig(config).items():
                    raw[k] = v
                data[config.name] = pd.concat(
                    [data[config.name], raw], ignore_index=True
                )
            for log in self.logs:
                log.update(logger)
            if (
                patience is not None
                and i % patience == 0
                and save_path is not None
            ):
                self.data = dict(data)
                self.to_pickle(save_path)
        self.data = dict(data)
        print(self.data)
        if save_path is not None:
            self.to_pickle(save_path)

    def to_pickle(
        self, path: Optional[str or pathlib.Path] = None, **kwargs
    ) -> str:
        if path is None:
            path = os.path.join(mkdtemp(), f"{self.shortname}.pkl")
        elif os.path.isdir(path):
            path = os.path.join(path, f"{self.shortname}.pkl")
        pd.to_pickle(self.data, path, **kwargs)
        return str(path)

    def read_pickle(self, path: str or pathlib.Path, **kwargs):
        if os.path.isdir(path):
            path = os.path.join(path, f"{self.shortname}.pkl")
        if not os.path.exists(path):
            return
        self.data = pd.read_pickle(path, **kwargs)


class ExpectationMetrics(Metrics):
    expectation_factory: Expectation = None

    def eval(self, config: Config):
        expectation = self.expectation_factory(
            config=config,
            num_workers=config.get("num_workers", 1),
            seed=config.get("seed", int(time.time())),
        )
        samples = expectation.compute(config.metrics.get("num_samples", 10))
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)

        stats = {}
        for k, v in sample_dict.items():
            stats[k] = Statistics.from_samples(
                v,
                reduction=config.metrics.get("reduction", "normal"),
                name=k,
            )

        stats = self.postprocess(stats)

        out = dict()
        for k, s in stats.items():
            if "samples" in s:
                out[k] = s.samples.tolist()
                continue
            for sk, sv in s.__data__.items():
                out[k + "_" + sk] = [sv]
        return out

    def postprocess(
        self, stats: Dict[str, Statistics]
    ) -> Dict[str, Statistics]:
        return stats


if __name__ == "__main__":
    pass
