import psutil
import numpy as np

from datetime import datetime, timedelta
from logging import Logger
from typing import Optional, Tuple


class MetricsLog:
    def __init__(self, name: str, logging_freq: int = 1):
        self.name = name
        self.logging_freq = logging_freq

    def setup(self) -> None:
        self.counter = 0

    def update(self, logger: Logger) -> None:
        raise NotImplementedError()


class ProgressLog(MetricsLog):
    def __init__(
        self, name: str, logging_freq: int = 1, total: Optional[int] = None
    ):
        super().__init__(self, name, logging_freq=logging_freq)
        self.total = total

    @staticmethod
    def timedelta_to_sec(dt: timedelta) -> int:
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
    def sec_to_timedelta(sec: int) -> timedelta:
        days, r = divmod(sec, 60 * 60 * 24)
        hours, r = divmod(r, 60 * 60)
        minutes, r = divmod(r, 60)
        seconds = r
        return timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        )

    def setup(self) -> None:
        super().setup(self)
        self.begin = datetime.now()

    def update(self, logger: Logger) -> None:
        self.counter += 1
        self.now = datetime.now()
        self.from_start = self.timedelta_to_second(self.now - self.begin)
        self.remaining = self.sec_to_timedelta(
            self.from_start * (self.total / self.counter) - 1
        )
        days, hours, mins, secs = self.timedelta_to_format(self.remaining)
        if self.counter % self.logging_freq == 0:
            msg = f"{self.name}: "
            msg += (
                f"{self.counter} / {self.total}"
                if self.total is not None
                else f"{self.counter}"
            )
            msg += f"---{days:0=2d}-{hours:0=2d}:{mins:0=2d}:{secs:0=2d}"
            logger.info(msg)


class MemoryLog(MetricsLog):
    def __init__(self, name: str, logging_freq: int = 1, unit: str = "gb"):
        super().__init__(self, name, logging_freq)
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

    def setup(self) -> None:
        super().setup(self)
        self.memory_usage = []

    def update(self, logger: Logger) -> None:
        self.memory_usage.append(
            round(psutil.virtual_memory().used / self.factor, 4)
        )
        if self.counter % self.logging_freq == 0:
            msg = f"{self.name}: {np.mean(self.memory_usage)} +- {np.std(self.memory_usage)}"
            logger.info(msg)
