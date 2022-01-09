import typing
from dataclasses import dataclass, field


@dataclass(order=True)
class Parameter:
    name: str
    value: typing.Any = None
    unique: bool = field(repr=False, default=False)
    force_non_sequence: bool = field(repr=False, default=False)

    @property
    def datatype(self) -> typing.Any:
        return self.infer_type(self.value)

    def __getitem__(self, key):
        if not self.is_sequenced():
            message = "this parameter is not sequenced."
            raise LookupError(message)
        return self.value[key]

    def set(self, value):
        self.value = value.value if issubclass(type(value), Parameter) else value

    def is_sequenced(self):
        return (
            issubclass(type(self.value), typing.Iterable)
            and not isinstance(self.value, str)
            and not self.force_non_sequence
        )

    def is_unique(self):
        return self.unique

    def infer_type(self, value: typing.Any):
        if self.force_non_sequence:
            return type(value)
        if isinstance(value, dict):
            message = "invalid value type `dict`."
            raise TypeError(message)
        if issubclass(type(value), typing.Iterable):
            return self.infer_type(next(value.__iter__()))
        return type(value)

    def generate_sequence(self):
        if self.is_sequenced():
            for v in self.value:
                yield v
        else:
            yield self.value


if __name__ == "__main__":
    pass
