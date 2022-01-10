from typing import Any, Callable, Dict

from .config import Config


def UnavailableOption(name):
    message = f"Option of name '{name}' are currently unavailable."
    raise NotImplementedError(message)


class OptionError(Exception):
    def __init__(self, actual: str = None, expected: str = None):
        if actual is None:
            return
        message = f"Option '{actual}' is invalid."
        if expected is not None:
            message = message[:-1]
            message += f", valid options are {expected}."
        super().__init__(message)


class MissingRequirementsError(Exception):
    def __init__(self, config: Config = None):
        if config is None:
            message = (
                f"Requirements `{config.unmet_requirements()}` of object "
                + f"`{config.__class__.__name__}` are missing and needs to be defined."
            )
        else:
            message = ""
        super().__init__(message)


class Factory:
    @classmethod
    def build(cls, config: Config) -> Any:
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        name = config.name
        if name in options:
            return options[name](config)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))


if __name__ == "__main__":
    pass