import os
import pathlib
import typing

from dataclasses import dataclass, field

from midynet.config import *

__all__ = ["ScriptManager"]


def split_into_chunks(
    container: typing.Iterable, num_chunks=None
) -> typing.Generator[typing.Iterable, None, None]:
    """Yield num_chunks number of sequential chunks from container."""
    num_chunks = 1 if num_chunks is None else num_chunks
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
    path_to_script: pathlib.Path = field(repr=False, default_factory=pathlib.Path)
    path_to_scripts: list[pathlib.Path] = field(repr=False, default_factory=list)
    modules_to_load: list[str] = field(repr=False, default_factory=list)
    env_to_load: str = field(repr=False, default=None)

    def __post_init__(self):
        self.exp = self.config.name
        self.config_array = [self.config]

    def write_script(self, config):
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

        script += self.executable

        for k, v in config.dict_copy().items():
            if v.is_config:
                continue
            if v.is_sequenced():
                script += f" --{k} {' '.join((str(vv) for vv in v.value))}"
            elif isinstance(v.value, (list, tuple)) and len(v.value) == 0:
                continue
            else:
                script += f" --{k} {v.value}"
        script += "\n \n"
        if self.env_to_load is not None:
            script += "deactivate\n"
        return script

    def set_up_script(self, config):
        script = self.write_script(config)
        path = self.path_to_script / f"{int(time.time())}.sh"
        with path.open("w") as f:
            f.write(script)
        return path

    def tear_down_script(self, path):
        path.unlink()

    def run(self):
        for c in self.config_array:
            path = self.set_up_script(c)
            path = os.system(f"{self.execution_command} {path}")
            self.tear_down_script(path)

    def split_param(self, key: str, num_chunks: int = None):
        new_config_array = []
        for c in self.config_array:
            p = c.get_param(key)
            if not p.is_sequenced():
                new_config_array.append(c)
                continue
            for val in split_into_chunks(p.get_value(), num_chunks):
                new_config = c.deepcopy()
                new_config.set_value(key, val)
                new_config_array.append(new_config)
        self.config_array = new_config_array


if __name__ == "__main__":
    pass
