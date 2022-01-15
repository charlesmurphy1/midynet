import os
import pathlib
import typing
import time

from dataclasses import dataclass, field

from midynet.config import *

__all__ = ["ScriptManager"]


def split_into_chunks(
    container: typing.Iterable, num_chunks=None
) -> typing.Generator[typing.Iterable, None, None]:
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
    name: str
    executable: str
    exp: str = None
    config: Config = field(repr=False, default_factory=Config)
    config_array: list[Config] = field(repr=False, default_factory=list)
    execution_command: str = field(repr=False, default="bash")
    resources: dict[str, str] = field(repr=False, default_factory=dict)
    resource_prefix: str = field(repr=False, default="#SBATCH")
    path_to_scripts: pathlib.Path = field(repr=False, default_factory=pathlib.Path)
    modules_to_load: list[str] = field(repr=False, default_factory=list)
    env_to_load: str = field(repr=False, default=None)

    def __post_init__(self):
        self.exp = self.config.name
        self.config_array = [self.config]
        if isinstance(self.path_to_scripts, str):
            self.path_to_scripts = pathlib.Path(self.path_to_scripts)
        if not self.path_to_scripts.exists():
            self.path_to_scripts.mkdir()

    def write_script(self, config: Config, tag: str):

        script = "#!/bin/bash\n"
        script += f"{self.resource_prefix} --job-name={self.name}\n"
        for k, r in self.resources.items():
            script += f"{self.resource_prefix} --{k}={r}\n"
        script += "\n"
        script += (
            f"module {' '.join(self.modules_to_load)}\n"
            if len(self.modules_to_load) > 0
            else ""
        )
        script += (
            f"source {self.env_to_load}\n  \n" if self.env_to_load is not None else ""
        )

        path_to_config = self.path_to_scripts / f"{tag}-config.pickle"
        script += f"{self.executable} {path_to_config}"

        script += "\n \n"
        if self.env_to_load is not None:
            script += "deactivate\n"
        return script

    def set_up(self, config: Config, tag: str = None) -> int:
        tag = f"{config.name}-{int(time.time())}" if tag is None else tag

        path_to_config = self.path_to_scripts / f"{tag}-config.pickle"
        with path_to_config.open("wb") as f:
            config.save(path_to_config)

        path_to_script = self.path_to_scripts / f"{tag}.sh"
        script = self.write_script(config, tag)
        with path_to_script.open("w") as f:
            f.write(script)
        return tag

    def tear_down(self, tag: int) -> None:
        path_to_script = self.path_to_scripts / f"{tag}.sh"
        path_to_config = self.path_to_scripts / f"{tag}-config.pickle"
        path_to_script.unlink()
        path_to_config.unlink()

    def run(self):
        for c in self.config_array:
            tag = self.set_up(c)
            path_to_script = self.path_to_scripts / f"{tag}.sh"
            os.system(f"{self.execution_command} {path_to_script}")
            self.tear_down(tag)

    def split_param(self, key: str, num_chunks: int = None):
        new_config_array = []
        for c in self.config_array:
            p = c.get_param(key)
            if not p.is_sequenced():
                new_config_array.append(c)
                continue
            for val in split_into_chunks(p.value, num_chunks):
                new_config = c.deepcopy()
                new_config.set_value(key, val)
                new_config_array.append(new_config)
        self.config_array = new_config_array


if __name__ == "__main__":
    pass
