import os
import pathlib
import time
from typing import Optional, Generator, Iterable
from dataclasses import dataclass, field

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


@dataclass
class ScriptManager:
    executable: str = field(repr=True, init=True)
    execution_command: str = field(repr=True, default="bash", init=True)
    path_to_scripts: pathlib.Path = field(
        repr=True, default_factory=pathlib.Path, init=True
    )
    path_to_log: Optional[pathlib.Path] = field(
        repr=True, default=None, init=True
    )

    def __post_init__(self):
        if isinstance(self.path_to_scripts, str):
            self.path_to_scripts = pathlib.Path(self.path_to_scripts)
        if not self.path_to_scripts.exists():
            self.path_to_scripts.mkdir(exist_ok=True, parents=True)

        if isinstance(self.path_to_log, str):
            self.path_to_log = pathlib.Path(self.path_to_log)
        if self.path_to_log is not None and not self.path_to_log.exists():
            self.path_to_log.mkdir(exist_ok=True, parents=True)

    def write_script(
        self,
        config: Config,
        run_name: Optional[str] = None,
        resource_prefix: str = "#SBATCH",
        nametag: str = "generic",
        modules_to_load: list[str] = None,
        virtualenv: str = None,
        extra_args: dict[str, str] = None,
    ):
        script = "#!/bin/bash\n"
        for k, r in (
            config.resources.dict.items()
            if config.resources is not None
            else dict().items()
        ):
            script += f"{resource_prefix} --{k}={r}\n"
        if self.path_to_log is not None:
            script += (
                f"{resource_prefix} --output="
                + f"{str(self.path_to_log)}/{config.name}.log\n"
            )

        script += "\n"
        if modules_to_load:
            script += f"module load {' '.join(modules_to_load)}\n"

        if virtualenv:
            script += f"source {virtualenv}\n  \n"

        path_to_config = self.path_to_scripts / f"{nametag}-config.pickle"
        script += f"{self.executable} --path_to_config {path_to_config}"
        if run_name is not None:
            script += f" --name {run_name.replace(' ', '_')}"

        extra_args = {} if extra_args is None else extra_args
        for k, v in extra_args.items():
            script += f" --{k} {v}"

        script += "\n \n"
        if virtualenv:
            script += "deactivate\n"
        return script

    def set_up(self, config: Config, **kwargs) -> int:

        nametag = f"{config.name}-{kwargs.pop('tag')}"
        path_to_config = self.path_to_scripts / f"{nametag}-config.pickle"
        with path_to_config.open("wb") as f:
            config.save(path_to_config)

        path_to_script = self.path_to_scripts / f"{nametag}.sh"
        script = self.write_script(config, nametag=nametag, **kwargs)
        with path_to_script.open("w") as f:
            f.write(script)
        return nametag

    def tear_down(self, nametag: int) -> None:
        path_to_script = self.path_to_scripts / f"{nametag}.sh"
        path_to_config = self.path_to_scripts / f"{nametag}-config.pickle"
        path_to_script.unlink()
        path_to_config.unlink()

    def run(self, config: Config, **kwargs):
        config = [config] if issubclass(config.__class__, Config) else config
        tag = kwargs.pop("tag") if "tag" in kwargs else int(time.time())

        for c in config:
            nametag = self.set_up(c, tag=tag, **kwargs)
            path_to_script = self.path_to_scripts / f"{nametag}.sh"
            os.system(f"{self.execution_command} {path_to_script}")
            tag += 1

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
