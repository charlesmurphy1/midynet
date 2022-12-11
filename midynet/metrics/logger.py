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
    def timedelta_to_second(dt: timedelta) -> int:
        days = dt.days
        hours, r = divmod(dt.seconds, 60 * 60)
        mins, r = divmod(r, 60)
        secs = r
        return ((days * 24 + hours) * 60 + mins) * 60 + secs

    @staticmethod
    def timedelta_to_format(dt: timedelta) -> Tuple[int, ...]:
        days = dt.days
        hours, r = divmod(dt.seconds, 60 * 60)
        mins, r = divmod(r, 60)
        secs = r
        return days, hours, mins, secs

    @staticmethod
    def second_to_timedelta(sec: int) -> timedelta:
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

    def update(self, logger: Optional[Logger] = None) -> None:
        self.counter += 1
        self.now = datetime.now()
        self.from_start = self.timedelta_to_second(self.now - self.begin)
        self.remaining = self.second_to_timedelta(
            self.from_start * ((self.total / self.counter) - 1)
        )
        days, hours, mins, secs = self.timedelta_to_format(self.remaining)
        if self.counter % self.logging_freq == 0 and logger is not None:
            msg = f"Progress: "
            msg += (
                f"{self.counter} / {self.total}"
                if self.total is not None
                else f"{self.counter}"
            )
            msg += f"\t time remaining : {days:0=2d}-{hours:0=2d}:{mins:0=2d}:{secs:0=2d}"
            if logger is not None:
                logger.info(msg)


class MemoryLog(MetricsLog):
    def __init__(self, logging_freq: int = 1, unit: str = "gb"):
        super().__init__(self)
        self.logging_freq = logging_freq
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
            msg = f"Memory: {np.mean(self.memory_usage)} +- {np.std(self.memory_usage)}"
            if logger is not None:
                logger.info(msg)
