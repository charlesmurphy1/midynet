import json
import pathlib

__all__ = ("Logger", "LoggerDict")


class Logger:
    def __init__(self):
        self.log = {}

    def on_task_begin(self):
        return

    def on_task_end(self):
        return

    def on_task_update(self, stepname=None):
        return

    def save(self, path: pathlib.Path):
        with path.open("w") as f:
            json.dump(self.log, f, indent=4)

    def load(self, path: pathlib.Path):
        with path.open("r") as f:
            self.log = json.load(f)


class LoggerDict:
    def __init__(self, **kwargs):
        self.loggers = kwargs

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return self.loggers[key]

    def keys(self):
        return self.loggers.keys()

    def values(self):
        return self.loggers.values()

    def items(self):
        return self.loggers.items()

    def on_task_begin(self):
        for logger in self.values():
            logger.on_task_begin()

    def on_task_end(self):
        for logger in self.values():
            logger.on_task_end()

    def on_task_update(self, stepname=None):
        for logger in self.values():
            logger.on_task_update(stepname)

    def save(self, path: pathlib.Path):
        log_dict = {}
        for k, l in self.items():
            log_dict[k] = l.log
        with path.open("w") as f:
            json.dump(log_dict, f, indent=4)

    def load(self, path: pathlib.Path):
        with path.open("r") as f:
            log_dict = json.load(f)
        for k, v in log_dict.items():
            if k in self:
                self[k].log = v


if __name__ == "__main__":
    pass
