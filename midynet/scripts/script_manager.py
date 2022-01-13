import pathlib

from dataclasses import dataclass, field

from midynet.config import Config

__all__ = []


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
class ScriptManage:
    executable: str
    name: str
    config: Config = field(repr=True, default_factory=Config)
    config_array: list[Config] = field(repr=False, default_factory=list)
    resources: list[str] = field(repr=False, default_factory=list)
    script_path: pathlib.Path = field(repr=False, default_factory=pathlib.Path)

    def __post_init__(self):
        self.config_array = [self.config]

    def write_script(self):
        pass

    def run_script(self):
        pass

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
