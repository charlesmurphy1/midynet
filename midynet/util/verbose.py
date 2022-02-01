import pathlib
import time
import typing
from datetime import datetime

import numpy as np
import tqdm

__all__ = ["Verbose"]


class Verbose:
    def __init__(
        self,
        filename: pathlib.Path = None,
        verbose_type: int = 0,
        progress_bar: typing.Callable = None,
    ):
        self.to_file = filename is not None
        self.filename = (
            pathlib.Path(filename) if isinstance(filename, str) else filename
        )
        self.verbose_type = verbose_type
        self.last_line = None
        self.reset()

        self.pb_template = tqdm.tqdm if progress_bar is None else progress_bar
        self.counter = 0

    def __call__(self, msg: str, overwrite_last: bool = False):
        if self.to_file:
            self.save_msg(msg, overwrite_last)
        if self.verbose_type != 0:
            self.print_msg(msg)
        self.last_line = msg

    def __repr__(self):
        return f"Verbose(type={self.verbose_type})"

    def save_msg(self, msg: str, overwrite_last: bool = False):
        if overwrite_last:
            with self.filename.open("r") as readf:
                lines = readf.readlines()
                lines[-1] = msg
            with self.filename.open("w") as writef:
                writef.writelines(lines)
                writef.close()
            return
        mode = "a" if self.filename.exists() else "w"
        with self.filename.open(mode) as writef:
            writef.write(f"{msg}\n")
            writef.close()

    def print_msg(self, msg: str):
        print(msg)

    def progress_msg(self, show_progress: bool = True, show_time: bool = True):
        msg = ""
        if show_progress:
            if self.total is not None:
                msg += f"\tProgress: {self.iteration} / {self.total}\t "
            else:
                msg += f"\tProgress: {self.iteration}\t "
        if show_time and self.iteration > 0:
            t = np.mean(self.times_per_step)
            total = t * self.total
            remaining = t * self.total - np.sum(self.times_per_step)
            msg += f" Time: {Verbose.format_time(total - remaining, short=True)} / {Verbose.format_time(total, short=True)}"

        if self.to_file:
            self.save_msg(msg, overwrite_last=self.iteration != 0)

    def update_progress(self):
        if self.iteration is not None:
            self.iteration += 1
            self.times_per_step.append(time.time() - self.prev)
            self.progress_msg()
            self.prev = time.time()

    def end_progress(self, show_time: bool = True):
        msg = ""
        if show_time:
            t = np.sum(self.times_per_step)
            msg += f"Total time {Verbose.format_time(t)}"
        self(msg, overwrite_last=True)
        self.reset()

    def init_progress(
        self, name: str, iterable: typing.Iterable = None, total: int = None
    ):
        self.pbar_name = name
        self.total = None
        self(name)
        if self.verbose_type == 1:
            if iterable is not None:
                pbar = self.pb_template(iterable, name, total=total)
                self.total = total
            elif total is not None:
                pbar = self.pb_template(range(total), name)
                self.total = total
            else:
                pbar = None
        else:
            pbar = iterable
            self.total = total
        self.prev = time.time()
        self.iteration = 0
        self.times_per_step = []

        self.progress_msg()

        return pbar

    def reset(self):
        self.iteration = None
        self.times_per_steps = None

    @staticmethod
    def format_time(t, short: bool = False):
        d = np.ceil(t // 60 // 60 // 24).astype("int")
        t -= d * 60 * 60 * 24
        h = np.ceil(t // 60 // 60).astype("int")
        t -= h * 60 * 60
        m = np.ceil(t // 60).astype("int")
        t -= m * 60
        s = np.ceil(t).astype("int")
        out = f"{d}-{h:0=2d}:{m:0=2d}:{s:0=2d}"
        if short and d == 0 and h == 0:
            out = f"{m:0=2d}:{s:0=2d}"
        elif short and d == 0 and h > 0:
            out = f"{h:0=2d}:{m:0=2d}:{s:0=2d}"
        return out


if __name__ == "__main__":
    pass
