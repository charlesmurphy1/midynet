import psutil
import numpy as np
import logging
import sys

from datetime import datetime, timedelta
from logging import Logger
from typing import Optional, Tuple


class MetricsLog:
    def __init__(self, logging_freq: int = 1):
        self.logging_freq = logging_freq

    def setup(self, **kwargs) -> None:
        self.counter = 0

    def update(self, logger: Logger) -> None:
        raise NotImplementedError()


class ProgressLog(MetricsLog):
    @staticmethod
    def timedelta_to_second(dt: timedelta) -> float:
        days = dt.days
        hours, r = divmod(dt.seconds, 60 * 60)
        mins, r = divmod(r, 60)
        secs = r + dt.microseconds / 1_000_000
        return ((days * 24 + hours) * 60 + mins) * 60 + secs

    @staticmethod
    def timedelta_to_format(dt: timedelta) -> Tuple[int, int, int, float]:
        days = dt.days
        hours, r = divmod(dt.seconds, 60 * 60)
        mins, r = divmod(r, 60)
        secs = r + dt.microseconds / 1_000_000
        return days, hours, mins, secs

    @staticmethod
    def second_to_timedelta(sec: float) -> timedelta:
        days, r = divmod(sec, 60 * 60 * 24)
        hours, r = divmod(r, 60 * 60)
        minutes, r = divmod(r, 60)
        seconds = r
        return timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        )

    def setup(self, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.counter = 0
        self.begin = datetime.now()
        self.last = datetime.now()

    def update(self, logger: Optional[Logger] = None) -> None:
        self.counter += 1
        self.now = datetime.now()
        self.from_start = self.timedelta_to_second(self.now - self.begin)
        self.from_last = self.now - self.last
        self.remaining = self.second_to_timedelta(
            self.from_start * ((self.total / self.counter) - 1)
        )
        if self.counter % self.logging_freq == 0 and logger is not None:
            msg = f"Progress: "
            msg += (
                f"{self.counter} / {self.total}"
                if self.total is not None
                else f"{self.counter}"
            )
            msg += f"\t elapsed : {self.now - self.begin}"
            msg += f"\t remaining : {self.remaining}"
            msg += f"\t per step : {self.from_last}"
            if logger is not None:
                logger.info(msg)
        self.last = self.now


class MemoryLog(MetricsLog):
    def __init__(
        self, logging_freq: int = 1, unit: str = "gb", round: int = -1
    ):
        super().__init__(self)
        self.logging_freq = logging_freq
        self.round = round
        if unit == "b":
            self.factor = 1
        elif unit == "kb":
            self.factor = 1024
        elif unit == "mb":
            self.factor = 1024**2
        elif unit == "gb":
            self.factor = 1024**3
        else:
            raise ValueError(
                f"`{unit}` is an invalid unit, valid units are"
                + "`[b, kb, mb, gb]`."
            )

    def setup(self, **kwargs) -> None:
        self.counter = 0
        self.memory_usage = []

    def update(self, logger: Optional[Logger]) -> None:
        self.memory_usage.append(
            round(psutil.virtual_memory().used / self.factor, 4)
        )
        if self.counter % self.logging_freq == 0 and logger is not None:
            msg = f"Memory: {np.mean(self.memory_usage): .4f} +- {np.std(self.memory_usage):.4f}"
            if logger is not None:
                logger.info(msg)
