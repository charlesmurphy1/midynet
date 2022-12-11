import os
import pathlib
import time
from typing import Optional, Generator, Iterable

from midynet.config import Config

__all__ = ["ScriptManager"]


def split_into_chunks(
    container: Iterable, num_chunks=None
) -> Generator[Iterable, None, None]:
    """Yield num_chunks number of sequential chunks from container."""
    if num_chunks is None:
        for c in container:
            yield c
        return
    d, r = divmod(len(container), num_chunks)
    for i in range(num_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield container[si : si + (d + 1 if i < r else d)]


class ScriptManager:
    def __init__(
        self,
        executable,
        execution_command: str = "bash",
        path_to_scripts=None,
    ):
        self.executable = executable
        self.execution_command = execution_command
        self.path_to_scripts = pathlib.Path(
            "./scripts" if path_to_scripts is None else path_to_scripts
        )
        if not self.path_to_scripts.exists():
            self.path_to_scripts.mkdir(exist_ok=True, parents=True)

    def write_script(
        self,
        resource_prefix: str = "#SBATCH",
        modules_to_load: list[str] = None,
        virtualenv: str = None,
        resources: dict[str, str] = None,
        extra_args: dict[str, str] = None,
    ):
        script = "#!/bin/bash\n"
        resources = {} if resources is None else resources
        for k, r in resources.items():
            script += f"{resource_prefix} --{k}={r}\n"

        script += "\n"
        if modules_to_load:
            script += f"module load {' '.join(modules_to_load)}\n"

        if virtualenv:
            script += f"source {virtualenv}\n  \n"
        script += f"{self.executable}"

        extra_args = {} if extra_args is None else extra_args
        for k, v in extra_args.items():
            script += f" --{k} {v}"

        script += "\n \n"
        if virtualenv:
            script += "deactivate\n"
        return script

    def clear(self) -> None:
        self.path_to_scripts.unlink()

    def run(self, name: str, **kwargs):
        path_to_script = self.path_to_scripts / f"{name}.sh"
        script = self.write_script(**kwargs)
        with path_to_script.open("w") as f:
            f.write(script)
        path_to_script = self.path_to_scripts / f"{name}.sh"
        os.system(f"{self.execution_command} {path_to_script}")

    # @staticmethod
    # def split_param(
    #     configs, param_key: str, num_chunks: int = None, label_with="value"
    # ):
    #     configs = (
    #         [configs] if issubclass(configs.__class__, Config) else configs
    #     )
    #     splitted_configs = []
    #     for c in configs:
    #         p = c.get_param(param_key)
    #         if not p.is_sequenced():
    #             splitted_configs.append(c)
    #             continue
    #         for i, val in enumerate(split_into_chunks(p.value, num_chunks)):
    #             new_config = c.deepcopy()
    #             new_config.set_value(param_key, val)
    #             if issubclass(val.__class__, Config):
    #                 ext = f"{val.name}"
    #             elif isinstance(val, str):
    #                 ext = f"{val}"
    #             else:
    #                 ext = f"{param_key.split('.')[-1]}{val}"
    #             if label_with == "value":
    #                 new_config.set_value("name", new_config.name + "." + ext)
    #             elif isinstance(label_with, list):
    #                 new_config.set_value(
    #                     "name", new_config.name + "." + label_with[i]
    #                 )
    #             else:
    #                 new_config.set_value("name", new_config.name + f".{i}")
    #             new_config.set_value(
    #                 "path", new_config.path / new_config.name
    #             )
    #             splitted_configs.append(new_config)
    #     return splitted_configs


if __name__ == "__main__":
    pass
