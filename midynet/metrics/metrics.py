import pathlib
import pandas as pd
import logging
import sys
import os
import numpy as np
import time
import logging
import multiprocessing as mp

from tempfile import mkdtemp
from typing import Dict, Optional
from collections import defaultdict

from midynet.config import Config
from midynet.metrics.logger import ProgressLog, MemoryLog
from .multiprocess import Expectation
from midynet.statistics import Statistics
from midynet.utility import to_batch

logger = logging.getLogger("midynet")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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

    def eval(
        self,
        config: Config,
        pool: Optional[mp.Pool] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def eval_async(
        self,
        config: Config,
        pool: Optional[mp.Pool] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def compute(
        self,
        configs: Config,
        logger: Optional[logging.Logger] = None,
        resume: bool = True,
        save_path: Optional[str] = None,
        patience: int = 1,
        num_workers: int = 1,
        num_async_jobs: int = 1,
    ) -> None:

        if save_path is not None and resume and os.path.exists(save_path):
            self.read_pickle(save_path)
        self.configs = configs
        config_seq = list(
            filter(self.already_computed, configs.to_sequence())
            if resume
            else configs.to_sequence()
        )
        self.logger = (
            logging.getLogger("midynet") if logger == "stdout" else logger
        )

        if num_async_jobs > 1 and num_workers > 1:
            data = self.run_async(
                config_seq, num_async_jobs, num_workers, patience, save_path
            )
        else:
            data = self.run(config_seq, num_workers, patience, save_path)

        self.data = dict(data)
        if save_path is not None:
            self.to_pickle(save_path)

    def run(
        self,
        config_seq: list[Config],
        num_workers: int = 1,
        patience: int = 1,
        save_path: Optional[str] = None,
    ):
        for log in self.logs:
            log.setup(total=len(config_seq))
        data = defaultdict(pd.DataFrame)
        data.update(self.data)
        for i, config in enumerate(config_seq):
            if num_workers > 1:
                with mp.get_context("spawn").Pool(num_workers) as p:
                    raw = pd.DataFrame(self.postprocess(self.eval(config, p)))
            else:
                raw = pd.DataFrame(self.postprocess(self.eval(config)))
            for k, v in self.configs.summarize_subconfig(config).items():
                raw[k] = v
            data[config.name] = pd.concat(
                [data[config.name], raw], ignore_index=True
            )
            for log in self.logs:
                log.update(self.logger)
            # saving current progress
            if (
                patience is not None
                and i % patience == 0
                and save_path is not None
            ):
                self.data = dict(data)
                self.to_pickle(save_path)

    def run_async(
        self,
        config_seq: list[Config],
        num_async_jobs: int,
        num_workers: int,
        patience: int = 1,
        save_path: Optional[str] = None,
    ):
        if num_workers == 1:
            raise ValueError("Cannot use async mode when num_workers == 1.")
        for log in self.logs:
            log.setup(
                total=len(config_seq) // num_async_jobs
                + int(len(config_seq) % num_async_jobs != 0)
            )
        data = defaultdict(pd.DataFrame)
        data.update(self.data)
        for i, batch in enumerate(to_batch(config_seq, num_async_jobs)):
            with mp.get_context("spawn").Pool(num_workers) as p:
                async_jobs = []

                # assign jobs
                for config in batch:
                    async_jobs.append(self.eval_async(config, p))

                # waiting for jobs to finish
                for job in async_jobs:
                    job.wait()

            # gathering results
            for job, config in zip(async_jobs, batch):
                raw = pd.DataFrame(self.postprocess(job.get()))
                for k, v in self.configs.summarize_subconfig(config).items():
                    raw[k] = v
                data[config.name] = pd.concat(
                    [data[config.name], raw], ignore_index=True
                )
            # saving current progress
            if (
                patience is not None
                and i % patience == 0
                and save_path is not None
            ):
                self.data = dict(data)
                self.to_pickle(save_path)
            for log in self.logs:
                log.update(self.logger)

        return data

    def postprocess(self, raw):
        return raw

    def already_computed(self, config):
        if config.name not in self.data:
            return False
        cond = pd.DataFrame()
        for k, v in self.configs.summarize_subconfig(config).items():
            cond[k] = self.data[config.name][k] == v
        return np.any(np.prod(cond.values, axis=-1))

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

    def eval(self, config: Config, pool: mp.Pool = None):
        expectation = self.expectation_factory(
            config=config,
            seed=config.get("seed", int(time.time())),
            num_samples=config.metrics.get("num_samples", 1),
        )
        return expectation.compute(pool)

    def eval_async(self, config: Config, pool: mp.Pool = None):
        expectation = self.expectation_factory(
            config=config,
            seed=config.get("seed", int(time.time())),
            num_samples=config.metrics.get("num_samples", 1),
        )
        return expectation.compute_async(pool)

    def reduce(
        self, samples: list[Dict[str, float]], reduction: str = "normal"
    ):
        return {
            k: Statistics.from_samples(
                [s[k] for s in samples], reduction=reduction, name=k
            )
            for k in samples[0]
        }

    def format(self, stats):
        out = dict()
        for k, s in stats.items():
            if "samples" in s:
                out[k] = s.samples.tolist()
                continue
            for sk, sv in s.__data__.items():
                out[k + "_" + sk] = [sv]
        return out

    def postprocess(
        self, samples: list[Dict[str, float]]
    ) -> Dict[str, Statistics]:
        return self.format(
            self.reduce(
                samples, self.configs.metrics.get("reduction", "normal")
            )
        )


if __name__ == "__main__":
    pass
